"""Fresh-process direct-load checkpoint dynamics for pending task P1.2."""
from __future__ import annotations
import argparse, csv, hashlib, json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

def _cell(args):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    from .config import load_config
    from .data import build_dataset
    from .evaluate import _trajectory_kl, accuracy_records
    from .hf_backend import encode_generation_prompts, load_tokenizer, model_init_kwargs, set_seed
    cfg=load_config(args.config); tokenizer=load_tokenizer(cfg,padding_side='left'); bundle=build_dataset(cfg.data,cfg.experiment.seed)
    # This forward pass occurs before PEFT is attached: genuinely unloaded base logits.
    model=AutoModelForCausalLM.from_pretrained(cfg.model.name_or_path,device_map='auto',**model_init_kwargs(cfg)); model.eval()
    encoded=encode_generation_prompts(tokenizer,[x.prompt for x in bundle.task_test[:2]],return_tensors='pt',padding=True,truncation=True,max_length=cfg.training.max_length).to(next(model.parameters()).device)
    with torch.inference_mode(): base_logits=model(**encoded,use_cache=False).logits.detach().float().cpu()
    fingerprint=hashlib.sha256(base_logits.numpy().tobytes()).hexdigest()
    model=PeftModel.from_pretrained(model,args.checkpoint,is_trainable=False); model.eval()
    set_seed(cfg.experiment.seed)
    task_records=accuracy_records(model,tokenizer,bundle.task_test,cfg,adapter=True)
    forward=[_trajectory_kl(model,tokenizer,x.prompt,cfg,source_adapter=False,direction='forward') for x in bundle.task_test[:cfg.evaluation.kl_samples]]
    payload={'task_records':task_records,'forward_kl_values':forward}
    result={'method':args.method,'step':int(Path(args.checkpoint).name.removeprefix('checkpoint-')),
            'checkpoint':args.checkpoint,'base_logits_sha256':fingerprint,'base_logits_shape':list(base_logits.shape),
            'task_accuracy':sum(float(x['correct']) for x in task_records)/len(task_records),'forward_kl':sum(forward)/len(forward),
            'evaluation':payload}
    path=Path(args.output); tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(result,indent=2),encoding='utf-8'); tmp.replace(path)
    print(json.dumps({k:result[k] for k in ('method','step','task_accuracy','forward_kl','base_logits_sha256')}),flush=True)

def _aggregate(args, cells):
    import matplotlib.pyplot as plt
    out=Path(args.output_dir); rows=[json.loads(p.read_text()) for p in cells]
    fingerprints={r['base_logits_sha256'] for r in rows}
    if len(fingerprints)!=1: raise RuntimeError(f'Unloaded base logits differed across fresh processes: {len(fingerprints)} fingerprints')
    selected={('sft',12),('grpo',20)}
    csv_rows=[{k:r[k] for k in ('method','step','checkpoint','task_accuracy','forward_kl','base_logits_sha256')} | {'selected_pair':(r['method'],r['step']) in selected} for r in rows]
    with (out/'p12_checkpoint_dynamics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(csv_rows[0])); w.writeheader(); w.writerows(csv_rows)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for method,color in (('sft','#1b9e77'),('grpo','#d95f02')):
        g=sorted((r for r in rows if r['method']==method),key=lambda r:r['step'])
        axes[0].plot([r['step'] for r in g],[r['task_accuracy'] for r in g],marker='o',label=method.upper(),color=color)
        axes[1].plot([r['step'] for r in g],[r['forward_kl'] for r in g],marker='o',label=method.upper(),color=color)
        axes[2].plot([r['forward_kl'] for r in g],[r['task_accuracy'] for r in g],marker='o',label=method.upper(),color=color)
        pick=[r for r in g if (method,r['step']) in selected]
        for r in pick:
            axes[0].scatter(r['step'],r['task_accuracy'],s=130,facecolors='none',edgecolors='black',linewidths=1.5)
            axes[1].scatter(r['step'],r['forward_kl'],s=130,facecolors='none',edgecolors='black',linewidths=1.5)
            axes[2].scatter(r['forward_kl'],r['task_accuracy'],s=130,facecolors='none',edgecolors='black',linewidths=1.5)
    axes[0].set(xlabel='Optimizer step',ylabel='GSM8K accuracy'); axes[1].set(xlabel='Optimizer step',ylabel='Forward KL from base'); axes[2].set(xlabel='Forward KL from base',ylabel='GSM8K accuracy')
    axes[0].legend(); fig.tight_layout()
    for ext in ('png','pdf'): fig.savefig(out/f'p12_checkpoint_dynamics.{ext}',dpi=200)
    summary={'checkpoint_count':len(rows),'sft_count':sum(r['method']=='sft' for r in rows),'grpo_count':sum(r['method']=='grpo' for r in rows),
             'fresh_process_per_checkpoint':True,'unloaded_base_logits_verified_identical':True,'base_logits_sha256':next(iter(fingerprints)),
             'selected_pair':{'sft_step':12,'grpo_step':20}}
    (out/'p12_checkpoint_dynamics_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps(summary,indent=2))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--sft-run'); p.add_argument('--grpo-run'); p.add_argument('--output-dir',required=True); p.add_argument('--checkpoint'); p.add_argument('--method'); p.add_argument('--output'); p.add_argument('--workers',type=int,default=3); a=p.parse_args()
    if a.checkpoint: return _cell(a)
    out=Path(a.output_dir); cells=[]; jobs=[]
    for method,run in (('sft',a.sft_run),('grpo',a.grpo_run)):
        checkpoints=sorted((p for p in (Path(run)/'checkpoints').glob('checkpoint-*') if (p/'adapter_model.safetensors').is_file()),key=lambda p:int(p.name.split('-')[-1]))
        for checkpoint in checkpoints:
            cell=out/f'p12_{method}_{checkpoint.name}.json'; cells.append(cell)
            if cell.is_file(): print(f'[P1.2] reuse {cell.name}',flush=True); continue
            cmd=[sys.executable,'-m','mats_experiments.checkpoint_dynamics','--config',a.config,'--output-dir',a.output_dir,'--checkpoint',str(checkpoint),'--method',method,'--output',str(cell)]
            jobs.append((method,checkpoint.name,cmd))
    def launch(job):
        method,name,cmd=job; print(f'[P1.2] fresh-load {method} {name}',flush=True); subprocess.run(cmd,check=True); return method,name
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(launch,job) for job in jobs]
        for future in as_completed(futures):
            method,name=future.result(); print(f'[P1.2] completed {method} {name}',flush=True)
    _aggregate(a,cells)
if __name__=='__main__': main()
