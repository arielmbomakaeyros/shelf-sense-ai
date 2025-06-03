import { NextResponse } from 'next/server';

export const runtime = 'edge'; // or 'nodejs' if you prefer

export async function POST(request: Request) {
  try {
    const formData = await request.formData();
    const file = formData.get('file') as File;

    // Replace with your FastAPI backend URL
    const backendUrl = 'http://localhost:8000/predict';
    
    const backendFormData = new FormData();
    backendFormData.append('file', file);

    const response = await fetch(backendUrl, {
      method: 'POST',
      body: backendFormData,
    });

    if (!response.ok) {
      throw new Error(await response.text());
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Unknown error' },
      { status: 500 }
    );
  }
}