'use client';

import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
} from '@/components/ui/card';
import { useStore } from '@/stores/data-store';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

export function MetricsDashboard() {
  const { predictions } = useStore();

  if (predictions.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Detection Metrics</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-gray-500">Upload an image to see detection metrics</p>
        </CardContent>
      </Card>
    );
  }

  // Count class occurrences
  const classCounts = predictions.reduce((acc, pred) => {
    acc[pred.class_name] = (acc[pred.class_name] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  const chartData = Object.entries(classCounts).map(([name, count]) => ({
    name,
    count,
  }));

  const averageConfidence = predictions.length > 0
    ? predictions.reduce((sum, pred) => sum + pred.confidence, 0) / predictions.length
    : 0;

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>Detection Metrics</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <h3 className="text-sm font-medium text-gray-500">Total Detections</h3>
              <p className="text-2xl font-bold">{predictions.length}</p>
            </div>
            <div>
              <h3 className="text-sm font-medium text-gray-500">Avg Confidence</h3>
              <p className="text-2xl font-bold">{(averageConfidence * 100).toFixed(1)}%</p>
            </div>
          </div>

          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData}>
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#3b82f6" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Detection Details</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            {predictions.map((pred, index) => (
              <div key={index} className="flex justify-between items-center p-2 border-b">
                <div>
                  <p className="font-medium">{pred.class_name}</p>
                  <p className="text-sm text-gray-500">
                    Confidence: {(pred.confidence * 100).toFixed(1)}%
                  </p>
                </div>
                <div className="text-sm text-gray-500">
                  {pred.bbox.map((coord, i) => (
                    <span key={i}>
                      {coord.toFixed(2)}
                      {i < 3 ? ', ' : ''}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}