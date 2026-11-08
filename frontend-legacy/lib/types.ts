export type Citation = {
  id: number | string;
  title: string;
  snippet: string;
  department?: string;
  source?: string;
  score?: number;
};

export type Report = {
  department: string;
  urgency: "normal" | "urgent" | "critical" | string;
  summary: string;
  advice: string[];
  medications: string[];
  citations: string[];
  confidence: number;
  followup_question: string;
};

export type Triage = {
  intent: string;
  department: string;
  urgency: string;
  symptoms: string;
  reason: string;
  confidence: number;
};

export type Safety = {
  requires_human_review: boolean;
  issues: string[];
  severity: string;
};

export type StreamEvent = {
  seq: number;
  type: string;
  data: Record<string, unknown>;
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  report?: Report;
  citations?: Citation[];
  safety?: Safety;
  triage?: Triage;
  error?: string;
};

export type Session = {
  id: string;
  user_id: string;
  title: string | null;
  created_at: string;
  updated_at: string;
};

export type ScheduleSlot = {
  slot_id: string;
  department: string;
  date: string;
  time: string;
  doctor: string;
  fee: number;
  available: boolean;
};

export type BookingOrder = {
  order_id: string;
  slot_id: string;
  patient_id: string;
  amount: number;
  status: string;
  payment_status: string;
  paid_at?: number;
  method?: string;
};

export type RagStatus = {
  documents: number;
  departments: { department: string; count: number }[];
  models: { opencode: boolean; siliconflow: boolean; ollama: boolean };
  mcp: boolean;
};

export type ModelCall = {
  provider: string;
  model: string;
  task: string;
  calls: number;
  prompt_tokens: number;
  completion_tokens: number;
  cost: number;
  avg_latency_ms: number;
};

export type EvalResult = {
  id: number;
  name: string;
  metrics: Record<string, unknown>;
  created_at: string;
};
