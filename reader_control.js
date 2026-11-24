"use strict";
// Fixed-query-count control. Node.js 18+, no dependencies.
function controlExperiment(){
  const D=32,T=16,H=24,seeds=[11,29,47,83,101],counts=[1,4,16],snaps=[16,64,256,512],lr=.025,testN=128;
  function rng(seed){let s=seed>>>0;return ()=>{s=(Math.imul(1664525,s)+1013904223)>>>0;return(s+.5)/4294967296}}
  function normal(r){return Math.sqrt(-2*Math.log(r()))*Math.cos(2*Math.PI*r())}
  function parity(x){let p=0;while(x){p^=1;x&=x-1}return p}
  function world(seed){let r=rng(seed),sign=Array.from({length:D},()=>r()<.5?-1:1),p=Array.from({length:D},(_,i)=>i);
    for(let i=D-1;i>0;i--){let j=Math.floor(r()*(i+1));[p[i],p[j]]=[p[j],p[i]]}
    let A=Array.from({length:D},(_,i)=>Float64Array.from({length:D},(_,j)=>sign[j]*(parity(i&p[j])?-1:1)/Math.sqrt(D)));
    function sample(){let y=Array.from({length:T},()=>r()<.5?-1:1),z=Array.from({length:D},(_,j)=>j<T?y[j]+.2*normal(r):normal(r)),x=Float64Array.from(A,row=>row.reduce((s,a,j)=>s+a*z[j],0));return{x,y}}
    return {train:Array.from({length:512},sample),test:Array.from({length:testN},sample)};
  }
  function model(mode,seed){let r=rng(seed),W=Float64Array.from({length:H*D},()=>normal(r)/Math.sqrt(D)),b=new Float64Array(H),Q=Float64Array.from({length:H*T},()=>normal(r)*.1),v=Float64Array.from({length:H},()=>normal(r)/Math.sqrt(H)),bias=new Float64Array(T),L=new Float64Array(T*D),hb=new Float64Array(H);let outBias=0;
    function pred(x,t,train,y){
      if(mode==="independent"){let s=bias[t],off=t*D;for(let j=0;j<D;j++)s+=L[off+j]*x[j];if(train){let e=-y/(1+Math.exp(Math.max(-60,Math.min(60,y*s))));for(let j=0;j<D;j++)L[off+j]-=lr*e*x[j];bias[t]-=lr*e}return s}
      let s=mode==="concat"?outBias:bias[t];
      for(let h=0;h<H;h++){let a=b[h]+(mode==="concat"?Q[h*T+t]:0),off=h*D;for(let j=0;j<D;j++)a+=W[off+j]*x[j];let z=Math.tanh(a);hb[h]=z;s+=z*(mode==="concat"?v[h]:Q[h*T+t]);}
      if(train){let e=-y/(1+Math.exp(Math.max(-60,Math.min(60,y*s))));for(let h=0;h<H;h++){let idx=h*T+t,z=hb[h],w=mode==="concat"?v[h]:Q[idx],g=e*w*(1-z*z),off=h*D;for(let j=0;j<D;j++)W[off+j]-=lr*g*x[j];b[h]-=lr*g;if(mode==="concat"){Q[idx]-=lr*g;v[h]-=lr*e*z}else Q[idx]-=lr*e*z;}if(mode==="concat")outBias-=lr*e;else bias[t]-=lr*e;}
      return s;
    }return pred;
  }
  let rows=[];
  for(const seed of seeds){let w=world(seed);
    for(const U of counts)for(const mode of ["concat","heads","independent"]){let pred=model(mode,seed+1000),r=rng(seed+3000),curve=[];
      for(let i=0;i<512;i++){let order=Array.from({length:T},(_,t)=>t);for(let j=T-1;j>0;j--){let l=Math.floor(r()*(j+1));[order[j],order[l]]=[order[l],order[j]]}for(const t of order)pred(w.train[i].x,t,true,w.train[i].y[t%U]);
        if(snaps.includes(i+1)){let right=0,wrong=0,per=Array(T).fill(0);for(const o of w.test)for(let t=0;t<T;t++){let y=o.y[t%U],ok=Number((pred(o.x,t,false,0)>=0?1:-1)===y);right+=ok;per[t]+=ok;wrong+=Number((pred(o.x,(t+1)%T,false,0)>=0?1:-1)===y);}
          curve.push({n:i+1,accuracy:right/(testN*T),shiftedQueryAccuracy:wrong/(testN*T),worst:Math.min(...per)/testN});}
      }rows.push({seed,U,mode,curve});
    }
  }
  const avg=x=>x.reduce((a,b)=>a+b,0)/x.length,summary=[];
  for(const U of counts)for(const mode of ["concat","heads","independent"]){let a=rows.filter(x=>x.U===U&&x.mode===mode);
    summary.push({U,mode,curve:snaps.map((n,i)=>({n,mean:avg(a.map(x=>x.curve[i].accuracy)),shifted:avg(a.map(x=>x.curve[i].shiftedQueryAccuracy)),worst:avg(a.map(x=>x.curve[i].worst)),minSeed:Math.min(...a.map(x=>x.curve[i].accuracy)),maxSeed:Math.max(...a.map(x=>x.curve[i].accuracy))}))});
  }
  return {protocol:{D,T,H,seeds,counts,snaps,lr,testN,primary:"fixed 16 query labels, vary distinct readout functions; matched 512 presentations per label",secondary:"wrong-query shift at held-out time",limitations:"Repeated functions receive more gradient presentations of the same underlying training objects; unique training objects, per-label exposure and total updates are matched, but per-function updates are not. Toy axes orthogonal, no biological evidence, architecture-specific."},rows,summary};
}
console.log(JSON.stringify(controlExperiment(),null,2));
