/**
 * Parameter Controls Component
 * 参数控制组件
 */

import React, { useState } from 'react';
import {
  Paper,
  Typography,
  Slider,
  Button,
  Box,
  Grid,
  TextField,
} from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';

export default function ParameterControls({ onComputeDOS, onGenerateTarget, loading }) {
  const [t_a, setT_a] = useState(-0.3);
  const [t_b, setT_b] = useState(-0.2);

  const handleComputeDOS = () => {
    onComputeDOS(t_a, t_b);
  };

  const handleGenerateTarget = () => {
    onGenerateTarget(t_a, t_b);
  };

  return (
    <Paper elevation={3} sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom>
        Hamiltonian Parameters
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Adjust tight-binding parameters for Kagome lattice
      </Typography>

      <Box sx={{ mb: 4 }}>
        <Typography gutterBottom>
          t_a (Nearest-neighbor hopping): {t_a.toFixed(3)}
        </Typography>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={10}>
            <Slider
              value={t_a}
              onChange={(e, newValue) => setT_a(newValue)}
              min={-0.5}
              max={0.5}
              step={0.01}
              marks={[
                { value: -0.5, label: '-0.5' },
                { value: 0, label: '0' },
                { value: 0.5, label: '0.5' },
              ]}
              disabled={loading}
            />
          </Grid>
          <Grid item xs={2}>
            <TextField
              size="small"
              type="number"
              value={t_a}
              onChange={(e) => setT_a(parseFloat(e.target.value))}
              inputProps={{ step: 0.01, min: -0.5, max: 0.5 }}
              disabled={loading}
            />
          </Grid>
        </Grid>
      </Box>

      <Box sx={{ mb: 4 }}>
        <Typography gutterBottom>
          t_b (Next-nearest-neighbor hopping): {t_b.toFixed(3)}
        </Typography>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={10}>
            <Slider
              value={t_b}
              onChange={(e, newValue) => setT_b(newValue)}
              min={-0.5}
              max={0.5}
              step={0.01}
              marks={[
                { value: -0.5, label: '-0.5' },
                { value: 0, label: '0' },
                { value: 0.5, label: '0.5' },
              ]}
              disabled={loading}
            />
          </Grid>
          <Grid item xs={2}>
            <TextField
              size="small"
              type="number"
              value={t_b}
              onChange={(e) => setT_b(parseFloat(e.target.value))}
              inputProps={{ step: 0.01, min: -0.5, max: 0.5 }}
              disabled={loading}
            />
          </Grid>
        </Grid>
      </Box>

      <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
        <Button
          variant="contained"
          color="primary"
          startIcon={<PlayArrowIcon />}
          onClick={handleComputeDOS}
          disabled={loading}
        >
          Compute DOS
        </Button>
        <Button
          variant="outlined"
          color="secondary"
          onClick={handleGenerateTarget}
          disabled={loading}
        >
          Set as Target
        </Button>
      </Box>

      <Box sx={{ mt: 2, p: 2, bgcolor: 'background.default', borderRadius: 1 }}>
        <Typography variant="caption" display="block" gutterBottom>
          <strong>Physical Meaning:</strong>
        </Typography>
        <Typography variant="caption" display="block" color="text.secondary">
          • t_a: Nearest-neighbor hopping amplitude
        </Typography>
        <Typography variant="caption" display="block" color="text.secondary">
          • t_b: Next-nearest-neighbor hopping amplitude
        </Typography>
        <Typography variant="caption" display="block" color="text.secondary">
          • Negative values indicate electron hopping between sites
        </Typography>
      </Box>
    </Paper>
  );
}
