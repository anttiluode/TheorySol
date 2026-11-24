"use strict";
// TheorySol fixed-map reader pilot. Node.js 18+, no dependencies.
function experiment() {
  const D=32,T=16, ks=[1,4,8,16], widths=[8,24], seeds=[11,29,47,83,101], snaps=[16,32,64,128,256], testN=128, lr=.025;
  function rng(seed){let s=seed>>>0;return ()=>{s=(Math.imul(1664525,s)+1013904223)>>>0;return (s+.5)/4294967296}}
  function normal(r){return Math.sqrt(-2*Math.log(r()))*Math.cos(2*Math.PI*r())}
  function parity(x){let p=0;while(x){p^=1;x&=x-1}return p}
  function world(seed){let r=rng(seed),sign=Array.from({length:D},()=>r()<.5?-1:1), perm=Array.from({length:D},(_,i)=>i);
    for(let i=D-1;i>0;i--){let j=Math.floor(r()*(i+1));[perm[i],perm[j]]=[perm[j],perm[i]]}
    let A=Array.from({length:D},(_,i)=>Float64Array.from({length:D},(_,j)=>sign[j]*(parity(i&perm[j])?-1:1)/Math.sqrt(D)));
    function sample(){let y=Array.from({length:T},()=>r()<.5?-1:1), z=Array.from({length:D},(_,j)=>j<T?y[j]+.20*normal(r):normal(r)), x=Float64Array.from(A,row=>row.reduce((s,a,j)=>s+a*z[j],0));return {x,y}}
    let train=Array.from({length:256},sample), test=Array.from({length:testN},sample);
    let oracle=0;for(const {x,y} of test)for(let t=0;t<T;t++){let z=0;for(let j=0;j<D;j++)z+=x[j]*A[j][t];oracle+=Number((z>=0?1:-1)===y[t])}
    if(oracle/(testN*T)<.99)throw Error("map sanity failed");
    return {train,test,oracle:oracle/(testN*T)}
  }
  function model(mode,H,seed){
    let r=rng(seed), W=Float64Array.from({length:H*D},()=>normal(r)/Math.sqrt(D)), b=new Float64Array(H), Q=Float64Array.from({length:H*T},()=>normal(r)*.1), v=Float64Array.from({length:H},()=>normal(r)/Math.sqrt(H)), bias=new Float64Array(T), L=new Float64Array(T*D), hb=new Float64Array(H);
    let outBias=0;
    function predict(x,t,train,y){
      if(mode==="independent"){let s=bias[t],off=t*D;for(let j=0;j<D;j++)s+=L[off+j]*x[j];if(train){let e=-y/(1+Math.exp(Math.max(-60,Math.min(60,y*s))));for(let j=0;j<D;j++)L[off+j]-=lr*e*x[j];bias[t]-=lr*e}return s}
      let s=mode==="concat"?outBias:bias[t];
      for(let h=0;h<H;h++){let a=b[h]+(mode==="concat"?Q[h*T+t]:0),off=h*D;for(let j=0;j<D;j++)a+=W[off+j]*x[j];let z=Math.tanh(a);hb[h]=z;s+=z*(mode==="concat"?v[h]:Q[h*T+t]);}
      if(train){let e=-y/(1+Math.exp(Math.max(-60,Math.min(60,y*s))));
        for(let h=0;h<H;h++){let idx=h*T+t,z=hb[h],w=mode==="concat"?v[h]:Q[idx],g=e*w*(1-z*z),off=h*D;
          for(let j=0;j<D;j++)W[off+j]-=lr*g*x[j];b[h]-=lr*g;
          if(mode==="concat"){Q[idx]-=lr*g;v[h]-=lr*e*z}else Q[idx]-=lr*e*z;
        }if(mode==="concat")outBias-=lr*e;else bias[t]-=lr*e;
      }return s;
    }
    return {predict,params:mode==="independent"?T*(D+1):H*D+H+H*T+(mode==="concat"?H+1:T)}
  }
  let rows=[];
  for(const seed of seeds){const w=world(seed);
    for(const K of ks){for(const H of widths){for(const mode of ["concat","heads","independent"]){if(mode==="independent"&&H!==widths[0])continue;
      const m=model(mode,H,seed+1000), orderR=rng(seed+3000);
      let curve=[];
      for(let i=0;i<256;i++){let order=Array.from({length:K},(_,t)=>t);
        for(let j=K-1;j>0;j--){let l=Math.floor(orderR()*(j+1));[order[j],order[l]]=[order[l],order[j]]}
        for(const t of order)m.predict(w.train[i].x,t,true,w.train[i].y[t]);
        if(snaps.includes(i+1)){let perTask=Array(K).fill(0);for(const o of w.test)for(let t=0;t<K;t++)perTask[t]+=Number((m.predict(o.x,t,false,0)>=0?1:-1)===o.y[t]);perTask=perTask.map(x=>x/testN);curve.push({examplesPerTask:i+1,totalPresentations:K*(i+1),accuracy:perTask.reduce((a,b)=>a+b,0)/K,worstTask:Math.min(...perTask),perTask});}
      }
      rows.push({seed,K,H:mode==="independent"?null:H,mode,params:m.params,oracle:w.oracle,curve});
    }}}
  }
  const mean=a=>a.reduce((x,y)=>x+y,0)/a.length;
  const summary=[];
  for(const K of ks)for(const H of widths)for(const mode of ["concat","heads","independent"]){if(mode==="independent"&&H!==widths[0])continue;const a=rows.filter(x=>x.K===K&&x.H===(mode==="independent"?null:H)&&x.mode===mode);
    const curve=snaps.map((n,i)=>({n,total:n*K,mean:mean(a.map(x=>x.curve[i].accuracy)),minSeed:Math.min(...a.map(x=>x.curve[i].accuracy)),maxSeed:Math.max(...a.map(x=>x.curve[i].accuracy)),meanWorstTask:mean(a.map(x=>x.curve[i].worstTask))}));
    summary.push({K,H:mode==="independent"?null:H,mode,params:a[0].params,curve,atMatched256:curve.find(x=>x.total===256),firstMean85:curve.find(x=>x.mean>=.85)?.n??null});
  }
  // Independent linear heads must produce exactly the same task-0 curve irrespective of K.
  for(const seed of seeds)for(const n of snaps){let a=rows.filter(x=>x.seed===seed&&x.mode==="independent").map(x=>x.curve.find(c=>c.examplesPerTask===n).perTask[0]);if(a.some(x=>x!==a[0]))throw Error("task exposure invariance failed")}
  return {protocol:{D,T,ks,widths,seeds,snaps,testN,lr,trainObjects:256,noiseSD:.2,primary:"matched per-task exposure, 256 examples per task",secondary:"matched total presentations, 256",threshold:.85,oracle:"known inverse of orthogonal map, unavailable to trained readers",notes:"Synthetic map frozen; all tasks see identical training objects in each seed; equal per-task exposure; fixed maximum query vocabulary; held-out objects; no tuning on held-out results. Descriptive five-seed pilot."},rows,summary};
}
console.log(JSON.stringify(experiment(), null, 2));
