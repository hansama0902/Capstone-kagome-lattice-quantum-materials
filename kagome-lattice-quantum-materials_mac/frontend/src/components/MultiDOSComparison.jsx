/**
 * Multi DOS Comparison Component
 * 多候选点DOS对比组件（类似原始代码的5个子图）
 */

import React, { useState, useEffect } from 'react';
import { 
  Paper, 
  Typography, 
  Box, 
  Button,
  CircularProgress,
  Alert,
} from '@mui/material';
import CompareArrowsIcon from '@mui/icons-material/CompareArrows';
import RefreshIcon from '@mui/icons-material/Refresh';

export default function MultiDOSComparison({ 
  targetDOS, 
  optimizationResults,
  onGenerateMultiPlot 
}) {
  const [loading, setLoading] = useState(false);
  const [imageData, setImageData] = useState(null);
  const [error, setError] = useState(null);

  const generatePlot = async () => {
    if (!targetDOS || !optimizationResults) {
      setError('Missing target DOS or optimization results');
      return;
    }

    setLoading(true);
    setError(null);
    
    try {
      // Prepare candidates data
      const candidates = optimizationResults.best_points.slice(0, 5).map((point, i) => ({
        point: point,
        rank: i + 1
      }));
      
      const data = await onGenerateMultiPlot(targetDOS.dos, targetDOS.bins, candidates);
      
      if (data && data.image) {
        setImageData(data);
      } else if (data && data.error) {
        setError(data.error);
      }
    } catch (err) {
      setError(err.message || 'Failed to generate plot');
    } finally {
      setLoading(false);
    }
  };

  if (!targetDOS || !optimizationResults) {
    return null;
  }

  return (
    <Paper elevation={3} sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <CompareArrowsIcon sx={{ mr: 1, fontSize: 28, color: 'secondary.main' }} />
          <Typography variant="h5" sx={{ fontWeight: 600 }}>
            Multi-Candidate DOS Comparison
          </Typography>
        </Box>
        <Button
          variant="contained"
          size="small"
          startIcon={<RefreshIcon />}
          onClick={generatePlot}
          disabled={loading}
          color="secondary"
        >
          Generate
        </Button>
      </Box>

      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Compare Target DOS with top 5 candidates (Target in Red, BO Suggested in Black, Final after local opt in Blue)
      </Typography>

      {loading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', p: 4 }}>
          <CircularProgress />
          <Typography variant="body2" sx={{ ml: 2 }}>
            Generating multi-panel comparison...
          </Typography>
        </Box>
      )}

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {imageData && imageData.image && !loading && (
        <Box>
          <Box 
            sx={{ 
              border: '2px solid',
              borderColor: 'divider',
              borderRadius: 2,
              overflow: 'hidden'
            }}
          >
            <img 
              src={`data:image/png;base64,${imageData.image}`}
              alt="Multi DOS Comparison"
              style={{ 
                width: '100%', 
                height: 'auto',
                display: 'block'
              }}
            />
          </Box>
          
          <Paper variant="outlined" sx={{ mt: 2, p: 2, bgcolor: 'grey.50' }}>
            <Typography variant="caption" display="block" sx={{ fontWeight: 600, mb: 1 }}>
              📊 Plot Guide
            </Typography>
            <Typography variant="caption" display="block" color="text.secondary">
              <span style={{ color: '#d32f2f', fontWeight: 'bold' }}>━━ Red:</span> Target DOS (true parameters)
            </Typography>
            <Typography variant="caption" display="block" color="text.secondary">
              <span style={{ color: '#000', fontWeight: 'bold' }}>━━ Black:</span> BO Suggested DOS (before local optimization)
            </Typography>
            <Typography variant="caption" display="block" color="text.secondary">
              <span style={{ color: '#1976d2', fontWeight: 'bold' }}>━━ Blue:</span> Final DOS (after local optimization refinement)
            </Typography>
          </Paper>
        </Box>
      )}
    </Paper>
  );
}
