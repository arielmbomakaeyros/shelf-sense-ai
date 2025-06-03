'use client';

import { useStore } from '@/stores/data-store';
// import { useStore } from '@/lib/store';
import { useEffect, useRef } from 'react';

export function PredictionViewer() {
  const { image, predictions } = useStore();
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!image || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const img = new Image();
    img.src = image;

    img.onload = () => {
      // Set canvas dimensions to match image
      canvas.width = img.width;
      canvas.height = img.height;

      // Draw image
      ctx.drawImage(img, 0, 0);

      // Draw bounding boxes
      predictions.forEach((pred) => {
        const [x1, y1, x2, y2] = pred.bbox;
        const width = x2 - x1;
        const height = y2 - y1;

        // Draw rectangle
        ctx.strokeStyle = '#3b82f6';
        ctx.lineWidth = 2;
        ctx.strokeRect(x1, y1, width, height);

        // Draw label background
        ctx.fillStyle = '#3b82f6';
        const text = `${pred.class_name} (${(pred.confidence * 100).toFixed(1)}%)`;
        const textWidth = ctx.measureText(text).width;
        ctx.fillRect(x1 - 1, y1 - 20, textWidth + 10, 20);

        // Draw text
        ctx.fillStyle = '#ffffff';
        ctx.font = '12px Arial';
        ctx.fillText(text, x1 + 4, y1 - 5);
      });
    };
  }, [image, predictions]);

  if (!image) return null;

  return (
    <div className="border rounded-lg overflow-hidden">
      <canvas ref={canvasRef} className="max-w-full h-auto" />
    </div>
  );
}