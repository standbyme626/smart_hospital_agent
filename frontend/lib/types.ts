export type User = { user: string; role: string };

export type Member = {
  id: string;
  user_id: string;
  name: string;
  relation: string;
  gender?: string | null;
  birth_date?: string | null;
  phone?: string | null;
  allergies?: string | null;
  conditions?: string | null;
  is_self: boolean;
  created_at?: string;
  updated_at?: string;
};

export type Citation = {
  id: number | string;
  doc_key?: string;
  title: string;
  snippet: string;
  department?: string;
  source?: string;
  score?: number;
};

export type DocumentDetail = {
  doc_key: string;
  title: string;
  content: string;
  department?: string | null;
  source?: string | null;
  metadata?: Record<string, unknown>;
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

export type Session = {
  id: string;
  user_id: string;
  title: string | null;
  created_at: string;
  updated_at: string;
};

export type Medication = {
  id: string;
  user_id: string;
  member_id?: string;
  name: string;
  dose?: string | null;
  frequency?: string | null;
  reminder_times?: string[];
  active: boolean;
  notes?: string | null;
  created_at?: string;
};

export type AbnormalValue = {
  name: string;
  value: string;
  reference: string;
  status: "high" | "low" | "normal" | string;
  explanation: string;
};

export type ReportRecord = {
  id: string;
  title: string;
  summary: string;
  abnormal_values: AbnormalValue[];
  questions: string[];
  created_at: string;
  image_path?: string;
};

export type Followup = {
  id: string;
  title: string;
  due_date: string | null;
  status: string;
  notes?: string | null;
  member_id?: string;
};

export type TimelineEvent = {
  id: number;
  kind: string;
  title: string;
  detail?: Record<string, unknown>;
  occurred_at: string | null;
};

export type HomeSummary = {
  member: Member;
  latest_session: Session | null;
  followups: Followup[];
  medications: Medication[];
  reports: ReportRecord[];
  timeline: TimelineEvent[];
};

export type Doctor = {
  doctor_id: string;
  name: string;
  title: string;
  hospital: string;
  department: string;
  rating: number;
  intro: string;
  fee: number;
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

export type VisitGuide = {
  order: BookingOrder;
  hospital: string;
  address: string;
  floor: string;
  checkin: string;
  items: string[];
  support: string;
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

export type ConsultPayload = {
  message: string;
  session_id?: string;
  member_id?: string;
};
