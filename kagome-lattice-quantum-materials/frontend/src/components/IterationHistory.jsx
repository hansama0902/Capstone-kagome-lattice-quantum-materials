/**
 * Iteration History Component
 * 迭代历史组件
 */

import React, { useState, useEffect } from 'react';
import {
  Paper,
  Typography,
  Box,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  Alert,
} from '@mui/material';
import HistoryIcon from '@mui/icons-material/History';
import axios from 'axios';

export default function IterationHistory({ onSelectIteration, targetParameters }) {
  const [history, setHistory] = useState([]);
  const [selectedIter, setSelectedIter] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchHistory();
    // Poll for updates every 5 seconds while optimization is running
    const interval = setInterval(fetchHistory, 5000);
    return () => clearInterval(interval);
  }, []);

  const fetchHistory = async () => {
    try {
      const response = await axios.get('http://localhost:5000/api/get_iteration_history');
      if (response.data.iterations && response.data.iterations.length > 0) {
        setHistory(response.data.iterations);
        // Auto-select latest iteration
        if (!selectedIter) {
          const latest = response.data.iterations[response.data.iterations.length - 1];
          setSelectedIter(latest.iteration);
        }
      }
    } catch (error) {
      console.error('Error fetching history:', error);
    }
  };

  const handleSelectIteration = (event) => {
    const iterNum = event.target.value;
    setSelectedIter(iterNum);
    
    const snapshot = history.find(h => h.iteration === iterNum);
    if (snapshot && onSelectIteration) {
      onSelectIteration(snapshot);
    }
  };

  const selectedSnapshot = history.find(h => h.iteration === selectedIter);

  if (history.length === 0) {
    return (
      <Paper elevation={3} sx={{ p: 3, textAlign: 'center' }}>
        <Typography variant="h6" color="text.secondary">
          Iteration History
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
          Run optimization to see iteration history
        </Typography>
      </Paper>
    );
  }

  return (
    <Paper elevation={3} sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
        <HistoryIcon sx={{ mr: 1 }} />
        <Typography variant="h6">
          Iteration History
        </Typography>
      </Box>

      <FormControl fullWidth sx={{ mb: 3 }}>
        <InputLabel>Select Iteration</InputLabel>
        <Select
          value={selectedIter}
          label="Select Iteration"
          onChange={handleSelectIteration}
        >
          {history.map((snapshot) => (
            <MenuItem key={snapshot.iteration} value={snapshot.iteration}>
              Iteration {snapshot.iteration} - {snapshot.n_evaluated} points evaluated
            </MenuItem>
          ))}
        </Select>
      </FormControl>

      {selectedSnapshot && (
        <Box>
          <Alert severity="info" sx={{ mb: 2 }}>
            <Typography variant="body2">
              <strong>Iteration {selectedSnapshot.iteration}</strong>
            </Typography>
            <Typography variant="caption" display="block">
              Total points evaluated: {selectedSnapshot.n_evaluated}
            </Typography>
          </Alert>

          {targetParameters && (
            <Box sx={{ mb: 2, p: 2, bgcolor: 'warning.light', borderRadius: 1 }}>
              <Typography variant="body2">
                <strong>True Parameters:</strong> t_a = {targetParameters.t_a.toFixed(3)}, 
                t_b = {targetParameters.t_b.toFixed(3)}
              </Typography>
            </Box>
          )}

          <TableContainer>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell><strong>Rank</strong></TableCell>
                  <TableCell><strong>t_a</strong></TableCell>
                  <TableCell><strong>t_b</strong></TableCell>
                  <TableCell><strong>Objective</strong></TableCell>
                  {targetParameters && <TableCell><strong>Error</strong></TableCell>}
                </TableRow>
              </TableHead>
              <TableBody>
                {selectedSnapshot.best_points.map((point, index) => {
                  const objective = selectedSnapshot.best_objectives[index][0];
                  const error_t_a = targetParameters ? Math.abs(point[0] - targetParameters.t_a) : null;
                  const error_t_b = targetParameters ? Math.abs(point[1] - targetParameters.t_b) : null;
                  
                  return (
                    <TableRow 
                      key={index}
                      sx={{ bgcolor: index === 0 ? 'action.selected' : 'inherit' }}
                    >
                      <TableCell>
                        {index === 0 ? (
                          <Chip label={`#${index + 1}`} color="primary" size="small" />
                        ) : (
                          `#${index + 1}`
                        )}
                      </TableCell>
                      <TableCell>{point[0].toFixed(4)}</TableCell>
                      <TableCell>{point[1].toFixed(4)}</TableCell>
                      <TableCell>
                        <Chip 
                          label={objective.toFixed(6)} 
                          size="small"
                          color={objective > -0.001 ? 'success' : 'default'}
                        />
                      </TableCell>
                      {targetParameters && (
                        <TableCell>
                          <Typography variant="caption">
                            Δt_a: {error_t_a.toFixed(4)}<br/>
                            Δt_b: {error_t_b.toFixed(4)}
                          </Typography>
                        </TableCell>
                      )}
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </TableContainer>
        </Box>
      )}
    </Paper>
  );
}
