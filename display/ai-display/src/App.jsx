import "./App.scss"

import {useState, useEffect} from 'react';

function App() {

  const [activeJobs, setActiveJobs] = useState([]);
  const [pastJobs, setPastJobs] = useState([]);
  const [showInfo, setShowInfo] = useState([]);

  useEffect(()=>{
    // get active jobs
    fetch(`${process.env.REACT_APP_BACKEND_URL}/api/active_jobs`)
      .then((res) => res.json())
      .then((data) => setActiveJobs(JSON.parse(data.data)));

    // get past jobs
    fetch(`${process.env.REACT_APP_BACKEND_URL}/api/past_jobs`)
    .then((res) => res.json())
    .then((data) => setPastJobs(JSON.parse(data.data)));


  },[]);

  useEffect(()=>{
    if (pastJobs.length > 0) console.log(pastJobs[0].model_code);
  }, [pastJobs])

  let show = false; 


  return (
    <div className="app">
      <div className="header">
        <h1>Display</h1>
      </div>
      <div className="runningJobs jobs">
        <h2>Active Jobs</h2>
        <div className="jobList">
          {activeJobs.length > 0 ? activeJobs.map((job, index) => (
            <div className="job" key={index}>
              <h3>{job.name}</h3>
              <p>STATUS: {job.status}</p>
              <p>IMAGE: {job.image}</p>
              <p>ID: {job.id}</p>
            </div>
          )) : null}
        </div>
      </div>
      <div className="pastJobs jobs">
      <h2>Past Jobs</h2>
        <div className="jobList">
          {pastJobs.length > 0 ? pastJobs.map((job, index) => (
            <div className="job" key={index} onClick={()=>showInfo.includes(index) 
              ? setShowInfo(showInfo.filter(item => item !== index)) 
              : setShowInfo([...showInfo, index])}>
              <h3>MODEL {job.model}</h3>
              <p>ACC: {job.accuracy}</p>
              <p>LOSS: {job.loss}</p>
              <p>TRAIN ACC: {job.train_acc}</p>
              <p>TRAIN LOSS: {job.train_loss}</p>
              <p>EPOCH: {job.epoch}</p>
              {showInfo.includes(index) ?  
              
              <div className="info">
                <div>
                  <pre>
                    <code>
                      {job.model_code}
                    </code>
                  </pre>
                </div>
                <div>
                  <pre>
                    <code>
                      {job.logs}
                    </code>
                  </pre>
                </div>
              </div>
              
              : null}
            </div>
          )) : null}
        </div>
      </div>

    </div>
  );
}

export default App;
