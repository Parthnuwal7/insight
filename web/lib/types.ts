/**
 * Types for the Insights frontend.
 *
 * These mirror the FastAPI backend contract in ABSA/app.py (verified
 * against the code, not the design doc). Every type here is the shape the
 * backend actually returns over JSON, sanitized server-side (non-finite
 * floats become null) so the browser never has to parse `NaN`.
 */

/** Discriminated result for every backend call. Functions never throw on
 * "the backend said no" — the interesting cases here are all specific. */
export type ApiResult<T> =
  | { ok: true; data: T }
  | { ok: false; status: number; message: string };

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------

export interface Health {
  status: "healthy" | "degraded" | "error";
  translation_service: "available" | "unavailable";
  absa_service: "available" | "unavailable";
  absa_error: string | null;
  keyword_fallback_enabled: boolean;
}

// ---------------------------------------------------------------------------
// Processing (POST /jobs, POST /process-reviews)
// ---------------------------------------------------------------------------

export interface ProcessRow {
  id: number;
  reviews_title: string;
  review: string;
  date: string;
  user_id: string;
}

export interface ProcessRequest {
  data: ProcessRow[];
  options?: Record<string, unknown>;
  user_id?: string;
}

export interface SubmitJobResponse {
  job_id: string;
}

export interface ProcessedRow {
  id: number;
  review: string;
  reviews_title: string;
  date: string;
  user_id: string;
  translated_review: string;
  detected_language: string;
  intent: string;
  intent_severity: string;
  intent_confidence: number;
  /** Surface forms, e.g. "Battery life" — show these to users. */
  aspects: string[];
  /** Normalised forms, e.g. "battery life" — group and count on these. */
  aspects_canonical: string[];
  /** Parallel to `aspects`: Positive | Negative | Neutral. Positional. */
  aspect_sentiments: string[];
  overall_sentiment: string;
  extraction_method: "pyabsa" | "none" | "llm_escalated" | string;
  degraded_reason: string | null;
}

export interface AspectRow {
  review_id: number;
  review: string;
  /** Surface form. */
  aspect: string;
  /** Canonical form — group and count on this, never the surface form. */
  aspect_canonical: string;
  aspect_sentiment: string;
  overall_sentiment: string;
  intent: string;
  intent_severity: string;
  date: string;
  language: string;
  extraction_method: string;
}

export interface MixedSentimentReview {
  review_id: number;
  review: string;
  aspects: string[];
  aspect_sentiments: string[];
  intent: string;
  date: string;
}

export interface ImprovementRow {
  aspect: string;
  negativity_pct: number;
  intent_severity: number;
  frequency: number;
  priority_score: number;
}

export interface StrengthAnchorRow {
  aspect: string;
  positivity_pct: number;
  intent_type: string;
  frequency: number;
  strength_score: number;
}

export interface SentimentAlert {
  aspect: string;
  spike_magnitude: number;
  recent_avg_negative: number;
  previous_avg_negative: number;
  alert_severity: "high" | "medium";
}

export interface Summary {
  total_reviews: number;
  total_aspects: number;
  mixed_sentiment_count: number;
  mixed_sentiment_pct: number;
  languages_detected: string[];
  intents_distribution: Record<string, number>;
  sentiment_distribution: Record<string, number>;
  top_problem_areas: number;
  top_strength_anchors: number;
  active_alerts: number;
}

export interface AspectNetwork {
  nodes: Array<{
    id: string;
    frequency?: number;
    sentiment_score?: number;
    color?: string;
  }>;
  links: Array<{ source: string; target: string; weight: number }>;
}

export interface JobResult {
  processed_data: ProcessedRow[];
  aspect_level_data: AspectRow[];
  mixed_sentiment_reviews: MixedSentimentReview[];
  areas_of_improvement: ImprovementRow[];
  strength_anchors: StrengthAnchorRow[];
  aspect_network: AspectNetwork | null;
  sentiment_alerts: SentimentAlert[];
  summary: Summary;
}

// ---------------------------------------------------------------------------
// Jobs (GET /jobs/{job_id})
// ---------------------------------------------------------------------------

export type JobStatusName =
  | "pending"
  | "running"
  | "completed"
  | "failed"
  | "cancelled";

export interface JobStatus {
  id: string;
  user_id: string;
  status: JobStatusName;
  stage: string | null;
  total_chunks: number;
  completed_chunks: number;
  error: string | null;
  created_at: number;
  updated_at: number;
}

export interface JobListResponse {
  jobs: JobStatus[];
}

// ---------------------------------------------------------------------------
// Report (POST /insights/report)
// ---------------------------------------------------------------------------

export type ClaimKind = "finding" | "complaint" | "strength" | "action";

export interface Claim {
  text: string;
  kind: ClaimKind;
  /** THE citation chain — resolve against processed_data to render. */
  review_ids: Array<number | string>;
  reason: string;
}

export interface VerificationStats {
  kept: number;
  dropped: number;
  dropped_by_reason: Record<string, number>;
}

export interface InvestigationStats {
  tool_calls_made: number;
  claims_returned: number;
  claims_discarded_uncited: number;
  claims_discarded_malformed: number;
  claims_deduplicated: number;
  stopped_reason: string;
  budget_hit: boolean;
  elapsed_seconds: number;
}

export interface ExtractionHealth {
  total: number;
  by_method: Record<string, number>;
  degraded_count: number;
  degraded_fraction: number;
}

export interface ClusterInfo {
  id: number;
  size: number;
  dominant_sentiment: string | null;
  top_aspects: string[];
}

export interface AspectStat {
  count: number;
  sentiment_counts: Record<string, number>;
  dominant_sentiment: string | null;
}

export interface EscalationStats {
  attempted: number;
  repaired: number;
  degraded: number;
  skipped: boolean;
  skipped_budget: number;
}

export interface ReportStats {
  claims_by_section: {
    findings: number;
    complaints: number;
    strengths: number;
    actions: number;
  };
  verification: VerificationStats;
  investigation: InvestigationStats;
  /** LLM-escalation bookkeeping. Present since the Phase C fix wave wired
   * escalation into the report path — treat as optional defensively. */
  escalation?: EscalationStats;
  extraction_health: ExtractionHealth;
  clusters: ClusterInfo[];
  aspect_stats: Record<string, AspectStat>;
}

export interface Report {
  findings: Claim[];
  complaints: Claim[];
  strengths: Claim[];
  actions: Claim[];
  stats: ReportStats;
  /** Render these prominently — they are facts about the run, not footnotes. */
  caveats: string[];
  /** True iff no section holds a claim. Not an error. */
  is_empty?: boolean;
}

export interface InsightReportRequest {
  processed_data: ProcessedRow[];
  aspect_level_data?: AspectRow[] | null;
  max_tool_calls?: number | null;
  max_seconds?: number | null;
}

export interface InsightReportResponse {
  status: "success";
  data: Report;
}
