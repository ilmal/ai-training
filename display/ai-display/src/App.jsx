import "./App.scss"

import {useState, useEffect} from 'react';

function App() {

  const [activeJobs, setActiveJobs] = useState([]);
  const [pastJobs, setPastJobs] = useState([]);

  useEffect(()=>{
    // get active jobs
    fetch(`${process.env.REACT_APP_BACKEND_URL}/api/active_jobs`)
      .then((res) => res.json())
      .then((data) => setActiveJobs(data.jobs));

    // get past jobs
    fetch(`${process.env.REACT_APP_BACKEND_URL}/api/past_jobs`)
    .then((res) => res.json())
    .then((data) => setActiveJobs(data.jobs));


  },[]);


  get_active_jobs = ()=>{

  }

  return (
    <div className="app">
      <div className="header">
        <h1>AI Display</h1>
      </div>
      <div className="runningJobs">
        
      </div>
      <div className="pastJobs">

      </div>

    </div>
  );
}

export default App;
