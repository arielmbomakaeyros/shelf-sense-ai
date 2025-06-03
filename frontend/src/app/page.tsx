import { ImageUploader } from '@/components/others/ImageUploader';
import { MetricsDashboard } from '@/components/others/MetricsDashboard';
import { PredictionViewer } from '@/components/others/PredictionViewer';
import { Card } from '@/components/ui/card';
// import { Card } from '@/components/ui/card';

export default function DashboardPage() {
  return (
    <div className="container mx-auto py-8">
      <h1 className="text-3xl font-bold mb-8">ShelfSense AI Dashboard</h1>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div className="space-y-8">
          <Card className="p-6">
            <h2 className="text-xl font-semibold mb-4">Upload Shelf Image</h2>
            <ImageUploader />
          </Card>

          <Card className="p-6">
            <h2 className="text-xl font-semibold mb-4">Detection Results</h2>
            <PredictionViewer />
          </Card>
        </div>

        <div>
          <MetricsDashboard />
        </div>
      </div>
    </div>
  );
}