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
import { Paper, Typography, Box } from '@mui/material';

export default function DOSComparison({ targetDOS, predictedDOS, title = "DOS Comparison" }) {
  if (!targetDOS || !predictedDOS) {
    return null;
  }

  // Prepare chart data
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
      <Typography variant="h6" gutterBottom>
        {title}
      </Typography>

      <Box sx={{ mb: 2, display: 'flex', justifyContent: 'space-between' }}>
        {targetDOS.true_parameters && (
          <Typography variant="body2" color="text.secondary">
            Target: t_a={targetDOS.true_parameters.t_a.toFixed(3)}, 
            t_b={targetDOS.true_parameters.t_b.toFixed(3)}
          </Typography>
        )}
        {predictedDOS.parameters && (
          <Typography variant="body2" color="text.secondary">
            Predicted: t_a={predictedDOS.parameters.t_a.toFixed(3)}, 
            t_b={predictedDOS.parameters.t_b.toFixed(3)}
          </Typography>
        )}
      </Box>

      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey="energy" 
            label={{ value: 'Energy (eV)', position: 'insideBottom', offset: -5 }}
            tick={{ fontSize: 11 }}
          />
          <YAxis 
            label={{ value: 'DOS (arb. units)', angle: -90, position: 'insideLeft' }}
            tick={{ fontSize: 11 }}
          />
          <Tooltip 
            formatter={(value) => value.toFixed(4)}
            labelFormatter={(label) => `Energy: ${label} eV`}
          />
          <Legend />
          <Line 
            type="monotone" 
            dataKey="target" 
            stroke="#d32f2f" 
            strokeWidth={2}
            dot={false}
            name="Target DOS"
          />
          <Line 
            type="monotone" 
            dataKey="predicted" 
            stroke="#1976d2" 
            strokeWidth={2}
            dot={false}
            name="Predicted DOS"
          />
        </LineChart>
      </ResponsiveContainer>

      <Box sx={{ mt: 2, p: 2, bgcolor: 'background.default', borderRadius: 1 }}>
        <Typography variant="caption" display="block">
          <strong>Error Metrics:</strong>
        </Typography>
        <Typography variant="caption" color="text.secondary">
          Mean Squared Error (MSE): {error.mse}
        </Typography>
      </Box>
    </Paper>
  );
}
