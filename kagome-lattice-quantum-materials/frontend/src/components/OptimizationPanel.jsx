/**
 * Optimization Panel Component
 * 优化控制面板组件
 */

import React, { useState, useEffect } from 'react';
import {
  Paper,
  Typography,
  Button,
  Box,
  LinearProgress,
  TextField,
  Grid,
  Alert,
  Chip,
} from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import StopIcon from '@mui/icons-material/Stop';
import SettingsIcon from '@mui/icons-material/Settings';

export default function OptimizationPanel({
  targetDOS,
  onStartOptimization,
  onStepOptimization,
  optimizationStatus,
  loading,
}) {
  const [nInitial, setNInitial] = useState(5);
  const [nIterations, setNIterations] = useState(10);
  const [isRunning, setIsRunning] = useState(false);

  const hasTarget = targetDOS && targetDOS.dos && targetDOS.bins;

  const handleStart = async () => {
    if (!hasTarget) {
      alert('Please generate a target DOS first!');
      return;
    }
    setIsRunning(true);
    await onStartOptimization(nInitial, nIterations);
  };

  const handleStep = async () => {
    await onStepOptimization();
  };

  const handleStop = () => {
    setIsRunning(false);
  };

  // Auto-update progress
  const progress = optimizationStatus?.current_iteration && optimizationStatus?.total_iterations
    ? (optimizationStatus.current_iteration / optimizationStatus.total_iterations) * 100
    : 0;

  return (
    <Paper elevation={3} sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
        <SettingsIcon sx={{ mr: 1 }} />
        <Typography variant="h6">
          Bayesian Optimization
        </Typography>
      </Box>

      {!hasTarget && (
        <Alert severity="info" sx={{ mb: 2 }}>
          Please generate a target DOS using the parameter controls first
        </Alert>
      )}

      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={6}>
          <TextField
            fullWidth
            label="Initial Points"
            type="number"
            value={nInitial}
            onChange={(e) => setNInitial(parseInt(e.target.value))}
            inputProps={{ min: 3, max: 20 }}
            disabled={loading || isRunning}
            size="small"
            helperText="Number of initial samples"
          />
        </Grid>
        <Grid item xs={6}>
          <TextField
            fullWidth
            label="Max Iterations"
            type="number"
            value={nIterations}
            onChange={(e) => setNIterations(parseInt(e.target.value))}
            inputProps={{ min: 5, max: 50 }}
            disabled={loading || isRunning}
            size="small"
            helperText="Number of BO iterations"
          />
        </Grid>
      </Grid>

      {optimizationStatus?.status && (
        <Box sx={{ mb: 2 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
            <Typography variant="body2">
              Status: <Chip 
                label={optimizationStatus.status} 
                size="small" 
                color={optimizationStatus.status === 'running' ? 'primary' : 'default'}
              />
            </Typography>
            {optimizationStatus.current_iteration !== undefined && (
              <Typography variant="body2">
                Iteration: {optimizationStatus.current_iteration} / {optimizationStatus.total_iterations}
              </Typography>
            )}
          </Box>
          {isRunning && <LinearProgress variant="determinate" value={progress} />}
        </Box>
      )}

      <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
        <Button
          variant="contained"
          color="primary"
          startIcon={<PlayArrowIcon />}
          onClick={handleStart}
          disabled={loading || isRunning || !hasTarget}
        >
          Start Optimization
        </Button>

        <Button
          variant="outlined"
          color="primary"
          onClick={handleStep}
          disabled={loading || !isRunning}
        >
          Step Once
        </Button>

        <Button
          variant="outlined"
          color="error"
          startIcon={<StopIcon />}
          onClick={handleStop}
          disabled={!isRunning}
        >
          Stop
        </Button>
      </Box>

      <Box sx={{ mt: 3, p: 2, bgcolor: 'background.default', borderRadius: 1 }}>
        <Typography variant="caption" display="block" gutterBottom>
          <strong>How it works:</strong>
        </Typography>
        <Typography variant="caption" display="block" color="text.secondary">
          1. Generate a target DOS with known parameters
        </Typography>
        <Typography variant="caption" display="block" color="text.secondary">
          2. BO explores parameter space intelligently
        </Typography>
        <Typography variant="caption" display="block" color="text.secondary">
          3. Finds parameters that best match the target
        </Typography>
        <Typography variant="caption" display="block" color="text.secondary">
          4. Uses Gaussian Process + Expected Improvement
        </Typography>
      </Box>
    </Paper>
  );
}
