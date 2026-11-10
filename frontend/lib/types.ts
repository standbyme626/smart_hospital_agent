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
  hospital_patient_id?: string | null;
  id_card_last4?: string | null;
  has_id_card?: boolean;
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
  member_id?: string | null;
  title: string | null;
  last_message_at?: string | null;
  created_at: string;
  updated_at: string;
};

export type ChatMessage = {
  id?: number;
  role: "user" | "assistant" | string;
  content: string;
  created_at?: string;
};

export type SessionDetail = {
  session: Session;
  messages: ChatMessage[];
  latest_final: {
    report?: Report;
    citations?: Citation[];
    triage?: Triage;
    safety?: Safety;
  } | null;
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
  has_image?: boolean;
  urgency?: string;
  critical_flags?: string[];
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

export type OrderRecord = {
  id: string;
  order_id?: string | null;
  department?: string | null;
  slot_id?: string | null;
  member_id?: string | null;
  hospital_patient_id?: string | null;
  status: string;
  payment_status: string;
  amount?: number;
  created_at?: string;
  updated_at?: string;
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

export type ReviewRequest = {
  id: string;
  kind: string;
  ref_id: string;
  user_id: string;
  member_id?: string | null;
  status: "pending" | "processing" | "approved" | "rejected" | "failed" | string;
  payload: Record<string, unknown>;
  decision_note?: string | null;
  decided_by?: string | null;
  decided_at?: string | null;
  created_at: string;
  department?: string | null;
  amount?: number | null;
  slot_id?: string | null;
  payment_status?: string | null;
  booking_status?: string | null;
  order_id?: string | null;
  username?: string | null;
  member_name?: string | null;
};

export type AdminUser = {
  id: string;
  username: string;
  role: "admin" | "patient" | string;
  enabled: boolean;
  created_at?: string;
  updated_at?: string;
};

export type AuditEntry = {
  request_id?: string | null;
  session_id?: string | null;
  actor?: string | null;
  action: string;
  detail: Record<string, unknown>;
  created_at: string;
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
  user_id?: string | null;
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

export type CatalogDepartment = {
  org_id: string;
  name: string;
};

export type CatalogDoctor = {
  practitioner_id: string;
  name: string;
  org_id: string;
};

export type CatalogService = {
  service_id: string;
  definition_id?: string | null;
  name: string;
  org_id: string;
  price: number;
};

export type BookingCatalog = {
  departments: CatalogDepartment[];
  doctors: CatalogDoctor[];
  services: CatalogService[];
  date?: string | null;
  availability_mode?: string;
  source?: string;
  stale?: boolean;
};

export type Appointment = {
  id: string;
  appointment_id?: string;
  user_id?: string;
  member_id?: string | null;
  his_patient_id?: string | null;
  quote_id?: string | null;
  amount: number;
  status: string;
  payment_status: string;
  his_status?: string | null;
  department?: string | null;
  org_id?: string | null;
  practitioner_id?: string | null;
  service_id?: string | null;
  appointment_date?: string | null;
  doctor_name?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
  stale?: boolean;
  legacy_demo?: boolean;
};

export type VisitStatus = {
  state: string;
  payment_status?: string | null;
  encounter_id?: string | null;
  guide?: string | null;
  next_action?: string | null;
  stale?: boolean;
};

export type AppointmentPayment = {
  payment_status: string;
  amount?: number | null;
  paid_at?: string | null;
  payment_id?: string | null;
  invoice_no?: string | null;
  stale?: boolean;
};

export type AppointmentDetail = {
  appointment: Appointment;
  visit: VisitStatus;
  payment: AppointmentPayment;
};

export type AppointmentCreateResult = {
  appointment: Appointment;
  instructions?: string;
  payment_mode?: string;
};

export type AppointmentCancelResult = {
  status: string;
  encounter_id?: string | null;
  refund_id?: string | null;
};

export type PaymentTask = {
  id?: string;
  task_id?: string | null;
  kind?: string;
  ref_id?: string;
  user_id?: string;
  member_id?: string | null;
  status: string;
  payload?: Record<string, unknown>;
  instructions?: string | null;
  sla_due_at?: string | null;
  assignee?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
};

export type Prescription = {
  his_prescription_no: string;
  status?: string | null;
  issued_at?: string | null;
  items?: unknown[] | null;
};

export type AppointmentGuide = {
  hospital: string;
  address: string;
  floor: string;
  checkin: string;
  items: string[];
  support: string;
  source?: string;
  stale?: boolean;
};

export type ConsultPayload = {
  message: string;
  session_id?: string;
  member_id?: string;
};
