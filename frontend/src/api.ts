import type { UiSearchRequest, UiSearchResponse, VideoSummaryResponse } from "./types";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || import.meta.env.VITE_API_URL || "").replace(/\/$/, "");

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
    ...init,
  });
  const body = await response.text();
  const payload = body ? JSON.parse(body) : null;
  if (!response.ok) {
    const detail = payload?.detail || response.statusText;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return payload as T;
}

export function search(request: UiSearchRequest): Promise<UiSearchResponse> {
  return requestJson<UiSearchResponse>("/api/v1/search", {
    method: "POST",
    body: JSON.stringify(request),
  });
}

export function getVideoSummary(videoId: string): Promise<VideoSummaryResponse> {
  return requestJson<VideoSummaryResponse>(`/api/v1/videos/${encodeURIComponent(videoId)}/summary`);
}

export function assetUrl(path: string): string {
  if (!path) {
    return "";
  }
  if (/^https?:\/\//.test(path)) {
    return path;
  }
  return `${API_BASE_URL}${path.startsWith("/") ? path : `/${path}`}`;
}
