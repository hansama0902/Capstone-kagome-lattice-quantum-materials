/**
 * Results Display Component
 * 结果展示组件
 */

import React from 'react';
import {
  Paper,
  Typography,
  Box,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  Button,
} from '@mui/material';
import EmojiEventsIcon from '@mui/icons-material/EmojiEvents';
import DownloadIcon from '@mui/icons-material/Download';

export default function ResultsDisplay({ 
  optimizationResults, 
  targetParameters,
  onLocalOptimize 
}) {
  if (!optimizationResults || !optimizationResults.best_points) {
    return (
      <Paper elevation={3} sx={{ p: 3, textAlign: 'center' }}>
        <Typography variant="h6" color="text.secondary">
          No Results Yet
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
          Run optimization to see results
        </Typography>
      </Paper>
    );
  }

  const { best_points, best_objectives } = optimizationResults;

  const handleLocalOptimize = (point, index) => {
    if (onLocalOptimize) {
      onLocalOptimize(point, index);
    }
  };

  // Calculate errors if true parameters are known
  const calculateError = (point) => {
    if (!targetParameters) return null;
    const error_t_a = Math.abs(point[0] - targetParameters.t_a);
    const error_t_b = Math.abs(point[1] - targetParameters.t_b);
    return { error_t_a, error_t_b };
  };

  return (
    <Paper elevation={3} sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
        <EmojiEventsIcon sx={{ mr: 1, fontSize: 32, color: 'warning.main' }} />
        <Typography variant="h5" sx={{ fontWeight: 600 }}>
          Best Candidates
        </Typography>
      </Box>

      {targetParameters && (
        <Box sx={{ mb: 3, p: 2, bgcolor: 'success.light', borderRadius: 1, border: '2px solid', borderColor: 'success.main' }}>
          <Typography variant="body1" sx={{ fontWeight: 600, color: 'success.dark' }}>
            🎯 True Parameters: t_a = {targetParameters.t_a.toFixed(3)}, t_b = {targetParameters.t_b.toFixed(3)}
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
              <TableCell><strong>Action</strong></TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {best_points.map((point, index) => {
              const objective = best_objectives[index][0];
              const error = calculateError(point);
              
              return (
                <TableRow 
                  key={index}
                  sx={{ 
                    bgcolor: index === 0 ? 'action.selected' : 'inherit',
                  }}
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
                  {targetParameters && error && (
                    <TableCell>
                      <Typography variant="caption">
                        Δt_a: {error.error_t_a.toFixed(4)}<br/>
                        Δt_b: {error.error_t_b.toFixed(4)}
                      </Typography>
                    </TableCell>
                  )}
                  <TableCell>
                    <Button
                      size="small"
                      variant="outlined"
                      onClick={() => handleLocalOptimize(point, index)}
                    >
                      Refine
                    </Button>
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </TableContainer>

      <Box sx={{ mt: 3, display: 'flex', justifyContent: 'space-between' }}>
        <Typography variant="caption" color="text.secondary">
          Higher objective values indicate better matches
        </Typography>
        <Button
          size="small"
          startIcon={<DownloadIcon />}
          variant="text"
        >
          Export Results
        </Button>
      </Box>
    </Paper>
  );
}
