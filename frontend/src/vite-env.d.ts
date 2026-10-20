/// <reference types="vite/client" />

interface Window {
  __astrixOrtSession?: import('onnxruntime-web').InferenceSession;
}
