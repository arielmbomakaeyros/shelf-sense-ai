import { create } from 'zustand';

interface Prediction {
  bbox: [number, number, number, number];
  confidence: number;
  class_id: number;
  class_name: string;
}

interface AppState {
  image: string | null;
  predictions: Prediction[];
  isLoading: boolean;
  error: string | null;
  uploadImage: (file: File) => Promise<void>;
  reset: () => void;
}

export const useStore = create<AppState>((set) => ({
  image: null,
  predictions: [],
  isLoading: false,
  error: null,
  uploadImage: async (file) => {
    set({ isLoading: true, error: null });
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(await response.text());
      }

      const data = await response.json();
      set({
        image: URL.createObjectURL(file),
        predictions: data.predictions,
        isLoading: false,
      });
    } catch (err) {
      set({
        error: err instanceof Error ? err.message : 'Unknown error occurred',
        isLoading: false,
      });
    }
  },
  reset: () => set({ image: null, predictions: [], error: null }),
}));