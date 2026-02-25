import streamlit as st
import json

def render_doctor_slots(slots_json: str):
    """
    Renders a list of doctor appointment slots as interactive cards.
    """
    try:
        slots = json.loads(slots_json)
        if not isinstance(slots, list):
            st.error("Invalid slots data format")
            return

        st.markdown("### 🏥 可预约医生 (Available Doctors)")
        
        # Use columns for grid layout
        cols = st.columns(2)
        
        for idx, slot in enumerate(slots):
            with cols[idx % 2]:
                with st.container(border=True):
                    st.markdown(f"**{slot['doctor']}**")
                    st.caption(f"📅 {slot['date']} | ⏰ {slot['time']}")
                    st.markdown(f"💰 挂号费: ¥{slot['fee']}")
                    
                    # Unique key for button
                    btn_key = f"book_{slot['slot_id']}"
                    
                    # Interactive button
                    if st.button("立即预约", key=btn_key, use_container_width=True):
                        # Add a message to session state to trigger backend
                        st.session_state.messages.append({
                            "role": "user",
                            "content": f"BOOK:{slot['slot_id']}"
                        })
                        st.rerun()

    except json.JSONDecodeError:
        st.error("Failed to decode slots data")

def render_payment_request(order_json: str):
    """
    Renders a payment confirmation card.
    """
    try:
        order = json.loads(order_json)
        st.markdown("### 💳 支付确认 (Payment Confirmation)")
        
        with st.container(border=True):
            st.info(f"订单号: {order.get('order_id')}")
            st.markdown(f"**金额**: ¥{order.get('payment_required') or order.get('amount')}")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("❌ 取消", key="cancel_pay"):
                    st.session_state.messages.append({"role": "user", "content": "Cancel payment"})
                    st.rerun()
            with col2:
                if st.button("✅ 确认支付", key="confirm_pay", type="primary"):
                    st.session_state.messages.append({
                        "role": "user", 
                        "content": f"PAY:{order.get('order_id')}"
                    })
                    st.rerun()
                    
    except Exception as e:
        st.error(f"Payment render error: {e}")
