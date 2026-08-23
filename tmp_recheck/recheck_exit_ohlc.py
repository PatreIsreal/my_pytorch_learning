#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import duckdb
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

KLINE_URL='https://huggingface.co/datasets/AlphaDojo/dojo_stock_kline/resolve/main/data.parquet?download=true'
TP=0.10
SL=0.15
MAX_OFFSET=3
COST=0.0025
FACTOR=(1-COST)/(1+COST)


def session():
    retry=Retry(total=8,connect=8,read=8,backoff_factor=2,status_forcelist=(429,500,502,503,504),allowed_methods=frozenset({'GET'}))
    s=requests.Session(); s.headers.update({'User-Agent':'best-strategy-ohlc-recheck/2026-08-23'}); s.mount('https://',HTTPAdapter(max_retries=retry)); return s

def download(url,path):
    if path.exists() and path.stat().st_size>1_000_000: return path.stat().st_size
    tmp=path.with_suffix('.part'); path.parent.mkdir(parents=True,exist_ok=True)
    with session().get(url,stream=True,timeout=(30,2400),allow_redirects=True) as r:
        r.raise_for_status()
        with tmp.open('wb') as f:
            for chunk in r.iter_content(8*1024*1024):
                if chunk: f.write(chunk)
    tmp.replace(path); return path.stat().st_size

def close(a,b,tol=1e-9):
    return abs(float(a)-float(b))<=tol*max(1.0,abs(float(a)),abs(float(b)))

def simulate(row,bars,market_dates):
    entry=pd.Timestamp(row.entry_date).date(); sym=row.symbol; entry_price=float(row.entry_price)
    target=entry_price*(1+TP); stop=entry_price*(1-SL)
    if entry not in market_dates: return {'error':'entry_not_market_date'}
    start=market_dates.index(entry)
    examined=[]
    for offset in range(MAX_OFFSET+1):
        d=market_dates[start+offset]
        key=(sym,d)
        if key not in bars:
            examined.append({'date':str(d),'missing':True})
            continue
        o,h,l,c=bars[key]
        examined.append({'date':str(d),'open':o,'high':h,'low':l,'close':c})
        if o<=stop:
            return {'exit_date':d,'exit_timing':'open','exit_price':o,'exit_reason':'gap_stop','holding_market_sessions':offset,'examined':examined}
        if o>=target:
            return {'exit_date':d,'exit_timing':'open','exit_price':o,'exit_reason':'gap_target','holding_market_sessions':offset,'examined':examined}
        hit_stop=l<=stop
        hit_target=h>=target
        if hit_stop:
            return {'exit_date':d,'exit_timing':'intraday','exit_price':stop,'exit_reason':'stop_loss','holding_market_sessions':offset,'examined':examined,'both_hit':bool(hit_target)}
        if hit_target:
            return {'exit_date':d,'exit_timing':'intraday','exit_price':target,'exit_reason':'take_profit','holding_market_sessions':offset,'examined':examined}
        if offset==MAX_OFFSET:
            return {'exit_date':d,'exit_timing':'close','exit_price':c,'exit_reason':'max_hold_close','holding_market_sessions':offset,'examined':examined}
    return {'error':'no_exit'}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',default='actual_39_legs.csv'); ap.add_argument('--data-dir',default='.tmp_recheck'); ap.add_argument('--out-dir',default='recheck_output'); args=ap.parse_args()
    inp=pd.read_csv(args.input,parse_dates=['signal_date','entry_date','exit_date'])
    data=Path(args.data_dir); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    kline=data/'kline.parquet'; size=download(KLINE_URL,kline)
    symbols=sorted(set(inp.symbol)|{'AAPL'})
    min_date=(inp.entry_date.min()-pd.Timedelta(days=7)).date(); max_date=(inp.exit_date.max()+pd.Timedelta(days=7)).date()
    con=duckdb.connect(); con.execute("PRAGMA threads=4"); con.execute("PRAGMA memory_limit='5GB'")
    quoted=','.join("'"+s.replace("'","''")+"'" for s in symbols)
    q=f"""
      WITH p AS (
        SELECT UPPER(REPLACE(CAST(symbol AS VARCHAR),'.','-')) symbol,
               TRY_CAST(bar_time AS DATE) trade_date,
               TRY_CAST(open AS DOUBLE) open, TRY_CAST(high AS DOUBLE) high,
               TRY_CAST(low AS DOUBLE) low, TRY_CAST(close AS DOUBLE) close,
               ROW_NUMBER() OVER(PARTITION BY UPPER(REPLACE(CAST(symbol AS VARCHAR),'.','-')),TRY_CAST(bar_time AS DATE) ORDER BY TRY_CAST(bar_time AS TIMESTAMP) DESC NULLS LAST) rn
        FROM read_parquet('{str(kline).replace("'","''")}')
        WHERE UPPER(COALESCE(CAST(kline_t AS VARCHAR),'1D'))='1D'
          AND UPPER(REPLACE(CAST(symbol AS VARCHAR),'.','-')) IN ({quoted})
          AND TRY_CAST(bar_time AS DATE) BETWEEN ? AND ?
      ) SELECT symbol,trade_date,open,high,low,close FROM p WHERE rn=1
    """
    df=con.execute(q,[min_date,max_date]).df(); df.trade_date=pd.to_datetime(df.trade_date).dt.date
    cal=sorted(df[df.symbol=='AAPL'].trade_date.unique().tolist())
    bars={(r.symbol,r.trade_date):(float(r.open),float(r.high),float(r.low),float(r.close)) for r in df.itertuples(index=False)}
    results=[]
    for r in inp.itertuples(index=False):
        sim=simulate(r,bars,cal)
        rec={'symbol':r.symbol,'entry_date':r.entry_date.date(),'reported_exit_date':r.exit_date.date(),'reported_exit_timing':r.exit_timing,'reported_exit_price':float(r.exit_price),'reported_exit_reason':r.exit_reason,'reported_holding':int(r.holding_market_sessions)}
        rec.update({k:v for k,v in sim.items() if k!='examined'})
        if 'error' not in sim:
            rec['entry_open_raw']=bars[(r.symbol,r.entry_date.date())][0]
            rec['entry_price_match']=close(rec['entry_open_raw'],r.entry_price,1e-10)
            rec['exit_date_match']=sim['exit_date']==r.exit_date.date()
            rec['exit_timing_match']=sim['exit_timing']==r.exit_timing
            rec['exit_reason_match']=sim['exit_reason']==r.exit_reason
            rec['exit_price_match']=close(sim['exit_price'],r.exit_price,1e-10)
            rec['holding_match']=int(sim['holding_market_sessions'])==int(r.holding_market_sessions)
            gross=float(sim['exit_price'])/float(r.entry_price)-1
            net=(1+gross)*FACTOR-1
            rec['gross_recomputed']=gross; rec['net_recomputed']=net
            rec['gross_match']=close(gross,r.gross_return,1e-10); rec['net_match']=close(net,r.net_return,1e-10)
            rec['all_match']=all(rec[x] for x in ['entry_price_match','exit_date_match','exit_timing_match','exit_reason_match','exit_price_match','holding_match','gross_match','net_match'])
            rec['examined_json']=json.dumps(sim.get('examined',[]),ensure_ascii=False)
        else: rec['all_match']=False
        results.append(rec)
    res=pd.DataFrame(results); res.to_csv(out/'raw_ohlc_exit_recheck.csv',index=False,encoding='utf-8-sig')
    summary={'generated_at_utc':pd.Timestamp.utcnow().isoformat(),'kline_size_bytes':size,'legs':len(res),'all_match_count':int(res.all_match.sum()),'failed_count':int((~res.all_match).sum()),'failed':res[~res.all_match].to_dict('records'),'both_hit_stop_first_count':int(res.get('both_hit',pd.Series(dtype=bool)).fillna(False).sum()),'entry_match_count':int(res.get('entry_price_match',pd.Series(dtype=bool)).fillna(False).sum()),'exit_date_match_count':int(res.get('exit_date_match',pd.Series(dtype=bool)).fillna(False).sum()),'exit_price_match_count':int(res.get('exit_price_match',pd.Series(dtype=bool)).fillna(False).sum()),'exit_reason_match_count':int(res.get('exit_reason_match',pd.Series(dtype=bool)).fillna(False).sum())}
    json.dump(summary,open(out/'raw_ohlc_exit_recheck_summary.json','w'),ensure_ascii=False,indent=2,default=str)
    print(json.dumps(summary,ensure_ascii=False,indent=2,default=str))
    if summary['failed_count']: raise SystemExit(2)
if __name__=='__main__': main()
