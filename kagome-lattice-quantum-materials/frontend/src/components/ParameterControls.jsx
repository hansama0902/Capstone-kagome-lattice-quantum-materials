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
  Chip,
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
      <Typography variant="h5" gutterBottom sx={{ fontWeight: 600, mb: 3 }}>
        ⚙️ Hamiltonian Parameters
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 4 }}>
        Adjust tight-binding parameters for Kagome lattice
      </Typography>

      <Box sx={{ mb: 4 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
          <Typography sx={{ fontWeight: 500 }}>
            t_a (Nearest-neighbor)
          </Typography>
          <Chip 
            label={t_a.toFixed(3)} 
            color="primary" 
            size="small"
            sx={{ fontWeight: 'bold', minWidth: 70 }}
          />
        </Box>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={9}>
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
              sx={{ '& .MuiSlider-markLabel': { fontSize: 11 } }}
            />
          </Grid>
          <Grid item xs={3}>
            <TextField
              size="small"
              type="number"
              value={t_a}
              onChange={(e) => setT_a(parseFloat(e.target.value) || 0)}
              inputProps={{ step: 0.01, min: -0.5, max: 0.5 }}
              disabled={loading}
              fullWidth
            />
          </Grid>
        </Grid>
      </Box>

      <Box sx={{ mb: 4 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
          <Typography sx={{ fontWeight: 500 }}>
            t_b (Next-nearest-neighbor)
          </Typography>
          <Chip 
            label={t_b.toFixed(3)} 
            color="secondary" 
            size="small"
            sx={{ fontWeight: 'bold', minWidth: 70 }}
          />
        </Box>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={9}>
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
              color="secondary"
              sx={{ '& .MuiSlider-markLabel': { fontSize: 11 } }}
            />
          </Grid>
          <Grid item xs={3}>
            <TextField
              size="small"
              type="number"
              value={t_b}
              onChange={(e) => setT_b(parseFloat(e.target.value) || 0)}
              inputProps={{ step: 0.01, min: -0.5, max: 0.5 }}
              disabled={loading}
              fullWidth
            />
          </Grid>
        </Grid>
      </Box>

      <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', mb: 3 }}>
        <Button
          variant="contained"
          color="primary"
          startIcon={<PlayArrowIcon />}
          onClick={handleComputeDOS}
          disabled={loading}
          size="large"
          sx={{ flex: 1, minWidth: 140 }}
        >
          Compute DOS
        </Button>
        <Button
          variant="contained"
          color="secondary"
          onClick={handleGenerateTarget}
          disabled={loading}
          size="large"
          sx={{ flex: 1, minWidth: 140 }}
        >
          Set as Target
        </Button>
      </Box>

      <Paper variant="outlined" sx={{ p: 2, bgcolor: 'info.lighter' }}>
        <Typography variant="caption" display="block" sx={{ fontWeight: 600, mb: 1 }}>
          📚 Physical Meaning
        </Typography>
        <Typography variant="caption" display="block" color="text.secondary" sx={{ lineHeight: 1.6 }}>
          <strong>t_a:</strong> Nearest-neighbor hopping amplitude
        </Typography>
        <Typography variant="caption" display="block" color="text.secondary" sx={{ lineHeight: 1.6 }}>
          <strong>t_b:</strong> Next-nearest-neighbor hopping amplitude
        </Typography>
        <Typography variant="caption" display="block" color="text.secondary" sx={{ lineHeight: 1.6 }}>
          Negative values indicate favorable electron hopping between lattice sites
        </Typography>
      </Paper>
    </Paper>
  );
}
