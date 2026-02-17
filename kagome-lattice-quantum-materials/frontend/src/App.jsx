/**
 * Main App Component
 * 主应用组件
 */

import React, { useState, useEffect } from 'react';
import {
  Container,
  Grid,
  AppBar,
  Toolbar,
  Typography,
  Box,
  Alert,
  Snackbar,
  CircularProgress,
  Chip,
} from '@mui/material';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import ScienceIcon from '@mui/icons-material/Science';

import kagomeAPI from './api/kagomeAPI';
import DOSVisualization from './components/DOSVisualization';
import DOSComparison from './components/DOSComparison';
import ParameterControls from './components/ParameterControls';
import OptimizationPanel from './components/OptimizationPanel';
import ResultsDisplay from './components/ResultsDisplay';
import ParameterSpaceViz from './components/ParameterSpaceViz';
import MultiDOSComparison from './components/MultiDOSComparison';

// Create theme
const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
});

function App() {
  // State management
  const [backendHealth, setBackendHealth] = useState(null);
  const [loading, setLoading] = useState(false);
  const [currentDOS, setCurrentDOS] = useState(null);
  const [targetDOS, setTargetDOS] = useState(null);
  const [optimizationResults, setOptimizationResults] = useState(null);
  const [optimizationStatus, setOptimizationStatus] = useState(null);
  const [parameterSpace, setParameterSpace] = useState(null);
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'info' });

  // Check backend health on mount
  useEffect(() => {
    checkBackendHealth();
  }, []);

  const checkBackendHealth = async () => {
    try {
      const health = await kagomeAPI.health();
      setBackendHealth(health);
      showSnackbar('Backend connected successfully', 'success');
    } catch (error) {
      console.error('Backend health check failed:', error);
      showSnackbar('Failed to connect to backend. Please start the server.', 'error');
    }
  };

  const showSnackbar = (message, severity = 'info') => {
    setSnackbar({ open: true, message, severity });
  };

  const handleCloseSnackbar = () => {
    setSnackbar({ ...snackbar, open: false });
  };

  // Compute DOS for current parameters
  const handleComputeDOS = async (t_a, t_b) => {
    setLoading(true);
    try {
      const result = await kagomeAPI.computeDOS(t_a, t_b);
      setCurrentDOS(result);
      showSnackbar(`DOS computed for t_a=${t_a.toFixed(3)}, t_b=${t_b.toFixed(3)}`, 'success');
    } catch (error) {
      console.error('Error computing DOS:', error);
      showSnackbar('Failed to compute DOS', 'error');
    } finally {
      setLoading(false);
    }
  };

  // Generate target DOS
  const handleGenerateTarget = async (t_a, t_b) => {
    setLoading(true);
    try {
      const result = await kagomeAPI.generateTarget(t_a, t_b);
      setTargetDOS(result);
      showSnackbar('Target DOS generated successfully', 'success');
    } catch (error) {
      console.error('Error generating target:', error);
      showSnackbar('Failed to generate target', 'error');
    } finally {
      setLoading(false);
    }
  };

  // Start Bayesian Optimization
  const handleStartOptimization = async (nInitial, nIterations) => {
    if (!targetDOS) {
      showSnackbar('Please generate a target DOS first', 'warning');
      return;
    }

    setLoading(true);
    try {
      const result = await kagomeAPI.startOptimization(
        targetDOS.dos,
        targetDOS.bins,
        nInitial,
        nIterations
      );
      setOptimizationResults(result);
      setOptimizationStatus({
        status: 'initialized',
        current_iteration: 0,
        total_iterations: nIterations,
      });
      showSnackbar(`Optimization initialized with ${nInitial} initial points`, 'success');
    } catch (error) {
      console.error('Error starting optimization:', error);
      showSnackbar('Failed to start optimization', 'error');
    } finally {
      setLoading(false);
    }
  };

  // Execute one optimization step
  const handleStepOptimization = async () => {
    setLoading(true);
    try {
      const result = await kagomeAPI.stepOptimization();
      
      if (result.status === 'completed') {
        showSnackbar('Optimization completed!', 'success');
        setOptimizationStatus({ ...optimizationStatus, status: 'completed' });
      } else {
        setOptimizationResults(result);
        setOptimizationStatus({
          status: 'running',
          current_iteration: result.current_iteration,
          total_iterations: result.total_iterations,
        });
        showSnackbar(`Iteration ${result.current_iteration} completed`, 'info');
      }
    } catch (error) {
      console.error('Error stepping optimization:', error);
      showSnackbar('Failed to step optimization', 'error');
    } finally {
      setLoading(false);
    }
  };

  // Local optimization
  const handleLocalOptimize = async (point, index) => {
    if (!targetDOS) return;

    setLoading(true);
    try {
      const result = await kagomeAPI.localOptimize(
        point,
        targetDOS.dos,
        targetDOS.bins
      );
      showSnackbar(
        `Local optimization: t_a=${result.optimized_point[0].toFixed(4)}, t_b=${result.optimized_point[1].toFixed(4)}`,
        'success'
      );
      
      // Compute DOS for optimized point
      const optimizedDOS = await kagomeAPI.computeDOS(
        result.optimized_point[0],
        result.optimized_point[1]
      );
      setCurrentDOS(optimizedDOS);
    } catch (error) {
      console.error('Error in local optimization:', error);
      showSnackbar('Failed to perform local optimization', 'error');
    } finally {
      setLoading(false);
    }
  };

  // Fetch parameter space visualization
  const handleFetchParameterSpace = async () => {
    if (!optimizationResults) {
      return null;
    }
    
    try {
      const data = await kagomeAPI.getParameterSpace(100);
      return data;
    } catch (error) {
      console.error('Error fetching parameter space:', error);
      return { error: error.message };
    }
  };

  // Generate multi-DOS comparison plot
  const handleGenerateMultiPlot = async (dos_target, bins_target, candidates) => {
    try {
      const data = await kagomeAPI.getMultiDOSPlot(dos_target, bins_target, candidates);
      return data;
    } catch (error) {
      console.error('Error generating multi DOS plot:', error);
      return { error: error.message };
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      
      {/* App Bar */}
      <AppBar position="static" elevation={2}>
        <Toolbar>
          <ScienceIcon sx={{ mr: 2 }} />
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Kagome Lattice Parameter Inversion
          </Typography>
          {backendHealth && (
            <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
              <Chip 
                label={backendHealth.device.toUpperCase()} 
                color="primary" 
                size="small" 
              />
              <Chip 
                label={backendHealth.backend} 
                variant="outlined" 
                size="small"
                sx={{ color: 'white', borderColor: 'white' }}
              />
            </Box>
          )}
        </Toolbar>
      </AppBar>

      {/* Main Content */}
      <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
        {!backendHealth && (
          <Alert severity="error" sx={{ mb: 3 }}>
            Cannot connect to backend server. Please start the server: 
            <code style={{ marginLeft: 8 }}>python app_pytorch.py</code>
          </Alert>
        )}

        {loading && (
          <Box sx={{ display: 'flex', justifyContent: 'center', mb: 2 }}>
            <CircularProgress />
          </Box>
        )}

        <Grid container spacing={3}>
          {/* Left Column - Controls */}
          <Grid item xs={12} md={4}>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
              <ParameterControls
                onComputeDOS={handleComputeDOS}
                onGenerateTarget={handleGenerateTarget}
                loading={loading}
              />
              
              <OptimizationPanel
                targetDOS={targetDOS}
                onStartOptimization={handleStartOptimization}
                onStepOptimization={handleStepOptimization}
                optimizationStatus={optimizationStatus}
                loading={loading}
              />
            </Box>
          </Grid>

          {/* Right Column - Visualizations */}
          <Grid item xs={12} md={8}>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
              {/* Target DOS */}
              {targetDOS && (
                <DOSVisualization 
                  dosData={targetDOS} 
                  title="Target DOS"
                />
              )}

              {/* Current DOS or Comparison */}
              {targetDOS && currentDOS ? (
                <DOSComparison
                  targetDOS={targetDOS}
                  predictedDOS={currentDOS}
                  title="DOS Comparison"
                />
              ) : currentDOS ? (
                <DOSVisualization 
                  dosData={currentDOS} 
                  title="Computed DOS"
                />
              ) : null}

              {/* Optimization Results */}
              {optimizationResults && (
                <ResultsDisplay
                  optimizationResults={optimizationResults}
                  targetParameters={targetDOS?.true_parameters}
                  onLocalOptimize={handleLocalOptimize}
                />
              )}

              {/* Multi-Candidate DOS Comparison */}
              {optimizationResults && targetDOS && (
                <MultiDOSComparison
                  targetDOS={targetDOS}
                  optimizationResults={optimizationResults}
                  onGenerateMultiPlot={handleGenerateMultiPlot}
                />
              )}

              {/* Parameter Space Visualization */}
              {optimizationResults && (
                <ParameterSpaceViz
                  optimizationResults={optimizationResults}
                  targetParameters={targetDOS?.true_parameters}
                  onFetchParameterSpace={handleFetchParameterSpace}
                />
              )}
            </Box>
          </Grid>
        </Grid>
      </Container>

      {/* Snackbar for notifications */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={4000}
        onClose={handleCloseSnackbar}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert 
          onClose={handleCloseSnackbar} 
          severity={snackbar.severity}
          sx={{ width: '100%' }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </ThemeProvider>
  );
}

export default App;
