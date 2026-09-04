"""Targeted E3 alpha=1 rerun for token windows found distinct by P2.2."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from .config import load_config
from .e4 import _collate_trajectories, _model_device, _sequence_kl
from .hf_backend import adapter_enabled, load_adapter_model, load_tokenizer
from .interventions import ResidualIntervention

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--sft-checkpoint',required=True); p.add_argument('--rl-checkpoint',required=True); p.add_argument('--raw-p0',required=True); p.add_argument('--raw-p12',required=True); p.add_argument('--output-dir',required=True); p.add_argument('--layer',type=int,default=10)
    a=p.parse_args()
    import torch
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    cfg=load_config(a.config); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    p0=torch.load(a.raw_p0,map_location='cpu',weights_only=True); p12=torch.load(a.raw_p12,map_location='cpu',weights_only=True)
    records=p0['p01_e3_trajectories']; boundary=p12['boundary']
    directions={w:m[:boundary].mean(0) for w,m in p12['sft_window_matrices'].items()}
    sft=load_adapter_model(cfg,a.sft_checkpoint); rl=load_adapter_model(cfg,a.rl_checkpoint); tok=load_tokenizer(cfg,padding_side='right')
    rows=[]
    for prompt_index,record in enumerate(records):
        ids,attn,mask=_collate_trajectories([record],tok.pad_token_id,_model_device(sft))
        with torch.inference_mode(), adapter_enabled(sft,True): source=sft(input_ids=ids,attention_mask=attn,use_cache=False).logits
        source_logp=F.log_softmax(source[:,:-1].float(),dim=-1)
        ids_rl=ids.to(_model_device(rl)); attn_rl=attn.to(_model_device(rl)); mask_rl=mask.to(_model_device(rl)); source_logp=source_logp.to(_model_device(rl))
        # Recompute each window independently; alpha is fixed at one.
        for window,direction in sorted(directions.items()):
            intervention=ResidualIntervention(a.layer,direction,'add',1.0)
            with intervention.install(rl):
                with torch.inference_mode(), intervention.disabled(), adapter_enabled(rl,True): baseline=rl(input_ids=ids_rl,attention_mask=attn_rl,use_cache=False).logits
                with torch.inference_mode(), adapter_enabled(rl,True): steered=rl(input_ids=ids_rl,attention_mask=attn_rl,use_cache=False).logits
            b=float(_sequence_kl(source_logp,baseline,mask_rl)[0].cpu()); s=float(_sequence_kl(source_logp,steered,mask_rl)[0].cpu())
            rows.append({'prompt_index':prompt_index,'token_window':window,'baseline_sft_to_rl_kl':b,'steered_sft_to_rl_kl':s,'delta_toward_sft_alpha1':b-s})
        print(f'[P2.2 E3] {prompt_index+1}/{len(records)}',flush=True)
    with (out/'p22_e3_alpha1.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary={}
    for window in sorted(directions):
        v=[r['delta_toward_sft_alpha1'] for r in rows if r['token_window']==window]
        summary[str(window)]={'mean_delta_toward_sft_alpha1':sum(v)/len(v),'values':v}
    (out/'p22_e3_alpha1_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    fig,ax=plt.subplots(figsize=(6,4)); windows=sorted(directions); means=[summary[str(w)]['mean_delta_toward_sft_alpha1'] for w in windows]
    ax.plot(windows,means,marker='o'); ax.axhline(0,color='black',linewidth=.7); ax.set(xlabel='Final prompt tokens used for direction',ylabel='Mean delta toward SFT (alpha=1)'); fig.tight_layout()
    for ext in ('png','pdf'): fig.savefig(out/f'p22_e3_alpha1.{ext}',dpi=200)
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
