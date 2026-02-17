/**
 * DOS Visualization Component
 * DOS可视化组件
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

export default function DOSVisualization({ dosData, title = "Density of States" }) {
  if (!dosData || !dosData.dos || !dosData.bins) {
    return (
      <Paper elevation={3} sx={{ p: 3, textAlign: 'center' }}>
        <Typography variant="h6" color="text.secondary">
          No DOS data available
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
          Generate target or compute DOS to visualize
        </Typography>
      </Paper>
    );
  }

  // Transform data for Recharts
  const chartData = dosData.dos.map((value, index) => ({
    energy: dosData.bins[index].toFixed(4),
    dos: value,
  }));

  return (
    <Paper elevation={3} sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom>
        {title}
      </Typography>
      
      {dosData.parameters && (
        <Box sx={{ mb: 2 }}>
          <Typography variant="body2" color="text.secondary">
            Parameters: t_a = {dosData.parameters.t_a.toFixed(3)}, 
            t_b = {dosData.parameters.t_b.toFixed(3)}
          </Typography>
        </Box>
      )}

      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey="energy" 
            label={{ value: 'Energy (eV)', position: 'insideBottom', offset: -5 }}
            tick={{ fontSize: 12 }}
          />
          <YAxis 
            label={{ value: 'DOS (arb. units)', angle: -90, position: 'insideLeft' }}
            tick={{ fontSize: 12 }}
          />
          <Tooltip 
            formatter={(value) => value.toFixed(4)}
            labelFormatter={(label) => `Energy: ${label} eV`}
          />
          <Legend />
          <Line 
            type="monotone" 
            dataKey="dos" 
            stroke="#1976d2" 
            strokeWidth={2}
            dot={false}
            name="DOS"
          />
        </LineChart>
      </ResponsiveContainer>

      <Box sx={{ mt: 2, display: 'flex', justifyContent: 'space-between' }}>
        <Typography variant="caption" color="text.secondary">
          Data points: {dosData.dos.length}
        </Typography>
        <Typography variant="caption" color="text.secondary">
          Energy range: [{dosData.bins[0].toFixed(3)}, {dosData.bins[dosData.bins.length - 1].toFixed(3)}] eV
        </Typography>
      </Box>
    </Paper>
  );
}
