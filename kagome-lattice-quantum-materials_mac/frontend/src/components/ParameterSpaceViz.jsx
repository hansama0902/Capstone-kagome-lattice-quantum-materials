/**
 * Parameter Space Visualization Component with Integrated History
 * 参数空间可视化组件（整合历史选择）
 */

import React, { useState, useEffect } from 'react';
import { 
  Paper, 
  Typography, 
  Box, 
  CircularProgress,
  Alert,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Grid,
} from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import TimelineIcon from '@mui/icons-material/Timeline';
import axios from 'axios';
import kagomeAPI from '../api/kagomeAPI';

export default function ParameterSpaceViz({ 
  optimizationResults, 
  onFetchParameterSpace 
}) {
  const [loading, setLoading] = useState(false);
  const [imageData, setImageData] = useState(null);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);
  const [selectedIter, setSelectedIter] = useState('current');

  // Fetch history periodically
  useEffect(() => {
    if (optimizationResults) {
      fetchHistory();
      const interval = setInterval(fetchHistory, 5001);
      return () => clearInterval(interval);
    }
  }, [optimizationResults]);

  // Auto-fetch image when optimization results change
  useEffect(() => {
    if (optimizationResults) {
      fetchImage();
    }
  }, [optimizationResults]);

  const fetchHistory = async () => {
    try {
      const response = await axios.get('http://localhost:5001/api/get_iteration_history');
      if (response.data.iterations) {
        setHistory(response.data.iterations);
      }
    } catch (error) {
      console.error('Error fetching history:', error);
    }
  };

  const fetchImage = async (iteration = null) => {
    setLoading(true);
    setError(null);
    
    try {
      // Pass iteration number to backend
      const iterNum = iteration || (selectedIter === 'current' ? null : selectedIter);
      
      // Call parent function with iteration parameter
      const data = await kagomeAPI.getParameterSpace(100, iterNum);
      
      if (data && data.image) {
        setImageData(data);
      } else if (data && data.error) {
        setError(data.error);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleIterationChange = (event) => {
    const iterValue = event.target.value;
    setSelectedIter(iterValue);
    
    // Fetch image for selected iteration
    if (iterValue === 'current') {
      fetchImage(null);  // Use current state
    } else {
      fetchImage(iterValue);  // Use historical snapshot
    }
  };

  if (!optimizationResults) {
    return (
      <Paper elevation={3} sx={{ p: 3, textAlign: 'center' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', mb: 2 }}>
          <TimelineIcon sx={{ mr: 1, fontSize: 28 }} />
          <Typography variant="h6">
            Parameter Space Exploration
          </Typography>
        </Box>
        <Typography variant="body2" color="text.secondary">
          Run optimization to visualize parameter space
        </Typography>
      </Paper>
    );
  }

  return (
    <Paper elevation={3} sx={{ p: 3 }}>
      {/* Header with controls */}
      <Grid container spacing={2} alignItems="center" sx={{ mb: 3 }}>
        <Grid item xs={12} md={6}>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <TimelineIcon sx={{ mr: 1, fontSize: 28, color: 'primary.main' }} />
            <Typography variant="h5" sx={{ fontWeight: 600 }}>
              Parameter Space Exploration
            </Typography>
          </Box>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Box sx={{ display: 'flex', gap: 2, justifyContent: 'flex-end' }}>
            <FormControl size="small" sx={{ minWidth: 220 }}>
              <InputLabel>View Iteration</InputLabel>
              <Select
                value={selectedIter}
                label="View Iteration"
                onChange={handleIterationChange}
              >
                <MenuItem value="current">
                  <strong>Current (Latest)</strong>
                </MenuItem>
                {history.map((snapshot) => (
                  <MenuItem key={snapshot.iteration} value={snapshot.iteration}>
                    Iteration {snapshot.iteration} • {snapshot.n_evaluated} points
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            
            <Button
              variant="outlined"
              size="small"
              startIcon={<RefreshIcon />}
              onClick={() => fetchImage()}
              disabled={loading}
            >
              Refresh
            </Button>
          </Box>
        </Grid>
      </Grid>

      {/* Status info */}
      {imageData && (
        <Alert severity="info" sx={{ mb: 2 }}>
          <Typography variant="body2">
            <strong>Viewing: Iteration {imageData.current_iteration}</strong> | 
            Points Evaluated: {imageData.n_points}
          </Typography>
        </Alert>
      )}

      {/* Loading state */}
      {loading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
          <CircularProgress />
          <Typography variant="body2" sx={{ ml: 2, alignSelf: 'center' }}>
            Generating visualization...
          </Typography>
        </Box>
      )}

      {/* Error state */}
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {/* Image display */}
      {imageData && imageData.image && !loading && (
        <Box>
          <Box 
            sx={{ 
              border: '2px solid',
              borderColor: 'divider',
              borderRadius: 2,
              overflow: 'hidden',
              mb: 2
            }}
          >
            <img 
              src={`data:image/png;base64,${imageData.image}`}
              alt="Parameter Space"
              style={{ 
                width: '100%', 
                height: 'auto',
                display: 'block'
              }}
            />
          </Box>
          
          {/* Legend */}
          <Paper variant="outlined" sx={{ p: 2, bgcolor: 'grey.50' }}>
            <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 600 }}>
              📊 Visualization Guide
            </Typography>
            
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <Typography variant="caption" display="block" sx={{ fontWeight: 600, mb: 0.5 }}>
                  Left Plot - Mean (Objective Value):
                </Typography>
                <Typography variant="caption" display="block" color="text.secondary">
                  • Bright regions (yellow) = Better parameter matches
                </Typography>
                <Typography variant="caption" display="block" color="text.secondary">
                  • Dark regions (pink) = Poorer matches
                </Typography>
                <Typography variant="caption" display="block" color="text.secondary">
                  • Contour lines = Constant objective value
                </Typography>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Typography variant="caption" display="block" sx={{ fontWeight: 600, mb: 0.5 }}>
                  Right Plot - Uncertainty:
                </Typography>
                <Typography variant="caption" display="block" color="text.secondary">
                  • Deep green = High uncertainty (unexplored)
                </Typography>
                <Typography variant="caption" display="block" color="text.secondary">
                  • Light color = Low uncertainty (well-explored)
                </Typography>
                <Typography variant="caption" display="block" color="text.secondary">
                  • Shows model confidence level
                </Typography>
              </Grid>
              
              <Grid item xs={12}>
                <Box sx={{ display: 'flex', gap: 3, flexWrap: 'wrap', mt: 1 }}>
                  <Typography variant="caption" sx={{ display: 'flex', alignItems: 'center' }}>
                    <Box component="span" sx={{ 
                      width: 12, height: 12, borderRadius: '50%', 
                      bgcolor: '#2196f3', mr: 0.5, border: '1px solid white'
                    }} />
                    Evaluated Points
                  </Typography>
                  <Typography variant="caption" sx={{ display: 'flex', alignItems: 'center' }}>
                    <Box component="span" sx={{ 
                      width: 12, height: 12, 
                      bgcolor: '#f44336', mr: 0.5, 
                      transform: 'rotate(45deg)', border: '1px solid white'
                    }} />
                    Top 5 Best
                  </Typography>
                  <Typography variant="caption" sx={{ display: 'flex', alignItems: 'center' }}>
                    <Box component="span" sx={{ 
                      fontSize: 16, color: 'gold', mr: 0.5
                    }}>★</Box>
                    True Parameters
                  </Typography>
                </Box>
              </Grid>
            </Grid>
          </Paper>
        </Box>
      )}
    </Paper>
  );
}
