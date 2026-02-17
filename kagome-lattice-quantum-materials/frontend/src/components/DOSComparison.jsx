/**
 * DOS Comparison Component
 * DOS对比组件
 */

import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { Paper, Typography, Box, Grid, Chip } from '@mui/material';

export default function DOSComparison({ targetDOS, predictedDOS, title = "DOS Comparison" }) {
  if (!targetDOS || !predictedDOS) {
    return null;
  }

  // Transform data for Recharts
  const chartData = targetDOS.dos.map((targetValue, index) => ({
    energy: targetDOS.bins[index].toFixed(4),
    target: targetValue,
    predicted: predictedDOS.dos[index] || 0,
  }));

  // Calculate error metrics
  const calculateError = () => {
    const target_norm = targetDOS.dos.map(v => v / (targetDOS.dos.reduce((a,b) => a+b, 0) + 1e-10));
    const pred_norm = predictedDOS.dos.map(v => v / (predictedDOS.dos.reduce((a,b) => a+b, 0) + 1e-10));
    
    const mse = target_norm.reduce((sum, val, i) => 
      sum + Math.pow(val - pred_norm[i], 2), 0) / target_norm.length;
    
    return { mse: mse.toFixed(8) };
  };

  const error = calculateError();

  return (
    <Paper elevation={3} sx={{ p: 3 }}>
      <Typography variant="h5" gutterBottom sx={{ fontWeight: 600, mb: 3 }}>
        🔬 {title}
      </Typography>

      {/* Parameters Info in Grid */}
      <Grid container spacing={2} sx={{ mb: 2 }}>
        <Grid item xs={12} md={6}>
          <Box sx={{ p: 1.5, bgcolor: 'error.light', borderRadius: 1 }}>
            <Typography variant="caption" display="block" sx={{ fontWeight: 'bold', color: 'error.dark' }}>
              Target Parameters
            </Typography>
            {targetDOS.true_parameters && (
              <Typography variant="body2" sx={{ mt: 0.5, color: 'error.dark' }}>
                t_a = {targetDOS.true_parameters.t_a.toFixed(3)}, 
                t_b = {targetDOS.true_parameters.t_b.toFixed(3)}
              </Typography>
            )}
          </Box>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Box sx={{ p: 1.5, bgcolor: 'info.light', borderRadius: 1 }}>
            <Typography variant="caption" display="block" sx={{ fontWeight: 'bold', color: 'info.dark' }}>
              Predicted Parameters
            </Typography>
            {predictedDOS.parameters && (
              <Typography variant="body2" sx={{ mt: 0.5, color: 'info.dark' }}>
                t_a = {predictedDOS.parameters.t_a.toFixed(3)}, 
                t_b = {predictedDOS.parameters.t_b.toFixed(3)}
              </Typography>
            )}
          </Box>
        </Grid>
      </Grid>

      <ResponsiveContainer width="100%" height={450}>
        <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 70 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey="energy" 
            label={{ value: 'Energy (eV)', position: 'insideBottom', offset: -55 }}
            tick={{ fontSize: 10, angle: -45, textAnchor: 'end' }}
            height={90}
          />
          <YAxis 
            label={{ value: 'DOS (arb. units)', angle: -90, position: 'insideLeft' }}
            tick={{ fontSize: 11 }}
            width={80}
          />
          <Tooltip 
            formatter={(value) => value.toFixed(4)}
            labelFormatter={(label) => `Energy: ${label} eV`}
          />
          <Legend 
            verticalAlign="top"
            height={36}
            wrapperStyle={{ paddingBottom: '15px' }}
            iconSize={16}
          />
          <Line 
            type="monotone" 
            dataKey="target" 
            stroke="#d32f2f" 
            strokeWidth={2.5}
            dot={false}
            name="Target DOS"
          />
          <Line 
            type="monotone" 
            dataKey="predicted" 
            stroke="#1976d2" 
            strokeWidth={2.5}
            dot={false}
            name="Predicted DOS"
          />
        </LineChart>
      </ResponsiveContainer>

      <Box sx={{ mt: 2, p: 2, bgcolor: 'background.default', borderRadius: 1 }}>
        <Grid container spacing={2}>
          <Grid item xs={12}>
            <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
              Error Metrics
            </Typography>
          </Grid>
          <Grid item xs={12} md={6}>
            <Chip 
              label={`MSE: ${error.mse}`} 
              color="primary" 
              variant="outlined"
              size="small"
            />
          </Grid>
        </Grid>
      </Box>
    </Paper>
  );
}
