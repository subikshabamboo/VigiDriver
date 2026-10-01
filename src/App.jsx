// App.jsx
import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import NavBar from './components/Navbar';
import VigiDriverSite from './components/VigiDriverSite';
import LiveMonitoringDashboard from './components/Livemonitoringdashboard';
import AnalyticsHistory from './components/Analyticshistory';
import ModelSettings from './components/Modelsettings';
import ModelSandbox from './components/ModelSandbox';

function App() {
  return (
    <Router>
      <Routes>
        {/* Landing page - no navbar */}
        <Route path="/" element={<VigiDriverSite />} />
        
        {/* Dashboard and other pages with navbar */}
        <Route path="/dashboard" element={
          <>
            <NavBar />
            <LiveMonitoringDashboard />
          </>
        } />
        <Route path="/analytics" element={
          <>
            <NavBar />
            <AnalyticsHistory />
          </>
        } />
        <Route path="/settings" element={
          <>
            <NavBar />
            <ModelSettings />
          </>
        } />
        <Route path="/sandbox" element={
          <>
            <NavBar />
            <ModelSandbox />
          </>
        } />
        
        {/* Redirect any unknown routes to dashboard */}
        <Route path="*" element={<Navigate to="/dashboard" replace />} />
      </Routes>
    </Router>
  );
}

export default App;