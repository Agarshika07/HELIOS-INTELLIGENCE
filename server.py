"""HELIOS INTELLIGENCE V1.2 — stateful mobility scenario simulator and decision console.
All generated telemetry is SIMULATION, not physical VAULT measurements.
"""
import json, threading, time
from collections import deque
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
import numpy as np
from helios.core import Helios, dataset, featurize, validate

WEB = Path(__file__).resolve().parent.parent / "web"
SCENARIOS = {
 "flat": {"label":"Flat terrain", "slope":1, "resistance":0.08, "vibration":0.025, "obstacle":False},
 "soft_sand": {"label":"Soft sand / traction loss", "slope":2, "resistance":0.95, "vibration":0.12, "obstacle":False},
 "deep_sand": {"label":"Deep soft sand / immobilisation", "slope":1, "resistance":1.0, "vibration":0.16, "obstacle":False},
 "steep_slope": {"label":"Steep slope", "slope":27, "resistance":0.48, "vibration":0.08, "obstacle":False},
 "uneven": {"label":"Uneven terrain", "slope":7, "resistance":0.28, "vibration":0.38, "obstacle":False},
 "obstacle": {"label":"Obstacle ahead", "slope":0, "resistance":0.12, "vibration":0.04, "obstacle":True},
 "wheel_block": {"label":"Wheel obstruction", "slope":0, "resistance":0.99, "vibration":0.22, "obstacle":False},
 "sensor_fault": {"label":"Sensor malfunction", "slope":0, "resistance":0.1, "vibration":0.04, "obstacle":False},
}

class Engine:
 def __init__(self):
  X,y,_=dataset(range(0,5)); self.model=Helios().fit(X,y); self.lock=threading.RLock(); self.reset('flat'); self.running=False
 def reset(self, scenario):
  if scenario not in SCENARIOS: raise ValueError('Unknown scenario')
  with self.lock:
   self.scenario=scenario; self.t=0.; self.x=0.; self.v=0.; self.heading=0.; self.command=0.65; self.wheel=0.; self.current=0.4
   self.rows=deque(maxlen=250); self.events=[]; self.log=[]; self.out=None; self.last_event=''; self.running=False
   self._event('Scenario loaded', 'SIMULATION initialized: '+SCENARIOS[scenario]['label'])
 def _event(self, title, detail):
  self.events.append({'time':round(self.t,1),'title':title,'detail':detail})
  self.events=self.events[-80:]
 def tick(self):
  with self.lock:
   p=SCENARIOS[self.scenario]; self.t+=0.2; dt=.2
   # scenario disturbances evolve with time; simple transparent educational physics model
   resistance=p['resistance']*(min(1,max(0,(self.t-2)/4)) if self.scenario in ('soft_sand','deep_sand','wheel_block') else 1)
   if self.scenario=='obstacle' and self.t>5: resistance=0.98
   if self.scenario=='sensor_fault' and self.t>4: resistance=.1
   slope_force=max(0,np.sin(np.deg2rad(p['slope'])))*0.85
   drive=self.command if self.running else 0
   traction=max(0,1-resistance)
   accel=(drive*traction - slope_force - .12*self.v) if drive else -0.8*self.v
   if self.scenario in ('deep_sand','wheel_block') and self.t>7: accel=min(accel,-.12)
   if self.scenario=='obstacle' and self.t>5: accel=-.8*self.v
   self.v=max(0,min(1.2,self.v+accel*dt)); self.x+=self.v*dt
   self.wheel=max(0,drive*(1+resistance*1.6)) if drive else 0
   self.current=(.35+drive*(.7+resistance*2.1)) if self.running else .12
   rng=np.random.default_rng(int(self.t*1000)+sum(map(ord,self.scenario)))
   vib=p['vibration']*(1+0.5*np.sin(self.t*3))
   impact=(.65 if self.scenario=='uneven' and int(self.t*2)%4==0 else 0)
   ax=float(accel+rng.normal(0,vib)+impact); ay=float(rng.normal(0,vib)); az=float(1+rng.normal(0,vib))
   gx=float(np.deg2rad(p['slope'])/10+rng.normal(0,vib)); gy=float(rng.normal(0,vib)); gz=float(rng.normal(0,vib))
   if self.scenario=='sensor_fault' and self.t>4: ax=18.0 if int(self.t*5)%2 else None
   row={'t':self.t,'ax':ax,'ay':ay,'az':az,'gx':gx,'gy':gy,'gz':gz,'speed':self.v,'wheel_speed':self.wheel,'current':self.current,'position':self.x}
   self.rows.append(row)
   # Infer from a complete 1-second window. Use the existing trained signal model.
   if len(self.rows)>=50:
    tail=list(self.rows)[-50:]
    w={k:np.asarray([r[k] for r in tail],dtype=float) for k in ('t','ax','ay','az','gx','gy','gz')}
    w['t']=np.arange(50)/50
    try:
     clean,q=validate(w); feats=featurize(clean)
     ml=self.model.infer(feats[-1],q) if len(feats) else {'state':'uncertain','confidence':0,'risk':0,'level':'CAUTION','alert':False,'anomaly':0,'action':'Verify sensors','explanation':'Insufficient signal window'}
    except Exception: q={'ok':False,'missing':1.,'saturated':0.,'irregular':0.}; ml={'state':'uncertain','confidence':0,'risk':0,'level':'CAUTION','alert':False,'anomaly':0,'action':'Verify sensors','explanation':'Signal window invalid'}
   else:
    q={'ok':True,'missing':0,'saturated':0,'irregular':0}; ml={'state':'warming up','confidence':0,'risk':0,'level':'NORMAL','alert':False,'anomaly':0,'action':'Collecting sensor window','explanation':'Waiting for sufficient sensor samples.'}
   commanded=self.running; progress_ratio=(self.v/max(self.command,0.1)) if commanded else 0
   discrepancy=commanded and self.wheel>0.25 and self.v<0.12 and self.t>5
   slope_risk=p['slope']>=20 and self.v<0.22 and commanded
   obstacle=p['obstacle'] and self.t>5
   stuck=bool(discrepancy and self.t>8 and self.scenario in ('soft_sand','deep_sand','wheel_block'))
   if not q['ok']: state='SENSOR DATA UNRELIABLE'; action='HOLD POSITION — inspect sensor'; prediction='State estimate may be unreliable'; risk='HIGH'
   elif obstacle: state='OBSTACLE BLOCKING PATH'; action='STOP — request path reassessment'; prediction='Continuing forward may cause collision'; risk='HIGH'
   elif stuck: state='STUCK SUSPECTED'; action='STOP — request recovery'; prediction='Current command is unlikely to restore progress'; risk='HIGH'
   elif slope_risk: state='SLOPE PROGRESS AT RISK'; action='HOLD POSITION — reassess slope'; prediction='Climbing progress may deteriorate'; risk='HIGH'
   elif commanded and self.v<.35 and self.t>4: state='MOVING WITH RESISTANCE'; action='REDUCE SPEED — monitor progress'; prediction='Movement may become immobilised if resistance persists'; risk='CAUTION'
   elif commanded and self.v>=.18: state='MOVING NORMALLY'; action='CONTINUE — monitor sensors'; prediction='Forward progress expected to continue under current simulated conditions'; risk='NORMAL'
   elif not commanded: state='STATIONARY (NO DRIVE COMMAND)'; action='WAIT — no movement requested'; prediction='Rover remains stationary while command is idle'; risk='NORMAL'
   else: state='STATIONARY'; action='VERIFY MOVEMENT'; prediction='Insufficient evidence to determine cause'; risk='CAUTION'
   out={'scenario':self.scenario,'scenario_label':p['label'],'source':'SIMULATION','time':round(self.t,1),'position_m':round(self.x,3),'speed_mps':round(self.v,3),'commanded':commanded,'command_pct':round(self.command*100 if commanded else 0,1),'wheel_speed_sim':round(self.wheel,3),'motor_current_sim':round(self.current,3),'slope_deg':p['slope'],'obstacle':obstacle,'state':state,'prediction':prediction,'action':action,'risk_level':risk,'progress_ratio':round(progress_ratio,3),'discrepancy':bool(discrepancy),'stuck_suspected':stuck,'quality':q,'ml':ml,'sensors':{'ax':ax,'ay':ay,'az':az,'gx':gx,'gy':gy,'gz':gz},'series':list(self.rows)[-100:]}
   self.out=out; self.log.append({k:out[k] for k in ('time','position_m','speed_mps','state','prediction','action','risk_level','ml')}); self.log=self.log[-2000:]
   eventkey=(state,action)
   if eventkey!=self.last_event: self.last_event=eventkey; self._event(state,action+' | '+prediction)
 def loop(self):
  while True:
   if self.running: self.tick()
   time.sleep(.2)
E=Engine()
class H(BaseHTTPRequestHandler):
 def sendj(self,code,obj,ctype='application/json'):
  b=obj if isinstance(obj,bytes) else json.dumps(obj,allow_nan=False).encode(); self.send_response(code); self.send_header('Content-Type',ctype); self.send_header('Content-Length',str(len(b))); self.send_header('Cache-Control','no-store'); self.end_headers(); self.wfile.write(b)
 def do_GET(self):
  p=self.path.split('?')[0]
  if p in ('/','/index.html'): return self.sendj(200,(WEB/'index.html').read_bytes(),'text/html; charset=utf-8')
  if p=='/api/health': return self.sendj(200,{'status':'ok','mode':'SIMULATION','scenarios':[{'id':k,'label':v['label']} for k,v in SCENARIOS.items()]})
  if p=='/api/state':
   with E.lock: return self.sendj(200,{'running':E.running,'scenario':E.scenario,'out':E.out,'events':E.events[-30:]})
  if p=='/api/export':
   with E.lock: return self.sendj(200,{'project':'HELIOS INTELLIGENCE V1.2','source':'SIMULATION','scenario':E.scenario,'events':E.events,'log':E.log})
  return self.sendj(404,{'error':'not found'})
 def do_POST(self):
  if self.path!='/api/control': return self.sendj(404,{'error':'not found'})
  try:
   d=json.loads(self.rfile.read(int(self.headers.get('Content-Length',0))) or b'{}'); a=d.get('action')
   if a=='start': E.running=True
   elif a=='pause': E.running=False
   elif a=='reset': E.reset(d.get('scenario',E.scenario))
   elif a=='scenario': E.reset(d['scenario'])
   elif a=='fault': E.scenario='sensor_fault'
   else: return self.sendj(400,{'error':'unknown action'})
   return self.sendj(200,{'ok':True,'running':E.running,'scenario':E.scenario})
  except Exception as ex: return self.sendj(400,{'error':str(ex)})
 def log_message(self,*a): pass

def main(port=8000):
 threading.Thread(target=E.loop,daemon=True).start(); print(f'HELIOS INTELLIGENCE V1.2: http://127.0.0.1:{port} (SIMULATION ONLY)'); ThreadingHTTPServer(('127.0.0.1',port),H).serve_forever()
if __name__=='__main__': main()
