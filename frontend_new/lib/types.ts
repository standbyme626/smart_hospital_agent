export type Role = "user" | "assistant";

export interface SlotItem {
  slot_id?: string;
  doctor?: string;
  date?: string;
  time?: string;
  fee?: number | string;
  department?: string;
  [key: string]: unknown;
}

export interface BookingPayload {
  slot_id?: string;
  order_id?: string;
  payment_required?: number | string;
  slot_info?: SlotItem;
  details?: Record<string, unknown>;
  message?: string;
  [key: string]: unknown;
}

export interface ChatMessage {
  id: string;
  role: Role;
  text: string;
  thoughts: string[];
  slots: SlotItem[];
}

export type StreamEventType =
  | "token"
  | "thought"
  | "status"
  | "ping"
  | "doctor_slots"
  | "booking_preview"
  | "payment_required"
  | "booking_confirmed"
  | "booking_error"
  | "tool_call"
  | "tool_output"
  | "phase"
  | "final"
  | "error";

export interface ParsedStreamEvent {
  type: StreamEventType;
  content: string;
  node?: string;
  requestId?: string;
  seq?: number;
  stage?: string;
  ts?: number;
  meta?: Record<string, unknown>;
  slots?: SlotItem[];
  booking?: BookingPayload;
}
