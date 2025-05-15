import React from "react";
import ProcessImages from "../components/ProcessImages";
import QTable from "../components/QTable";
import RunSimulation from "../components/RunSimulation";


function App() {
  return (
    <div className="max-w-5xl mx-auto mt-10">
      <h1 className="text-4xl font-bold text-center text-blue-600 mb-10">
        Traffic Simulation Dashboard
      </h1>
      <div className="space-y-10">
        <ProcessImages />
        <RunSimulation />
        <QTable />
      </div>
    </div>
  );
}

export default App;
