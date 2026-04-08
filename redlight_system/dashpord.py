import streamlit as st
import sqlite3
import pandas as pd
import os
import time
import subprocess

# 1. Page Configuration
st.set_page_config(page_title="Smart Traffic Hub", layout="wide", page_icon="🚦")

# 2. Professional Custom CSS
st.markdown("""
    <style>
    .main-title { font-size: 32px; font-weight: bold; color: #1e3a8a; margin-bottom: 5px; }
    .plate-display { 
        direction: ltr !important; unicode-bidi: bidi-override; 
        font-size: 35px; font-weight: bold; color: #1e3a8a; letter-spacing: 2px;
    }
    .violation-card { 
        background-color: #ffffff; border-radius: 15px; padding: 25px; 
        border-right: 8px solid #1e3a8a; box-shadow: 0 4px 12px rgba(0,0,0,0.05); 
        margin-bottom: 20px; 
    }
    .kpi-box { 
        background-color: #f0f4f8; padding: 15px; border-radius: 12px; 
        text-align: center; border: 1px solid #d1d5db;
    }
    .kpi-value { font-size: 20px; font-weight: bold; color: #1e3a8a; }
    </style>
    """, unsafe_allow_html=True)

def fetch_data():
    db_path = os.path.join("events", "violations.db")
    if not os.path.exists(db_path): 
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql("SELECT * FROM violations", conn)
        conn.close()
        return df
    except:
        return pd.DataFrame()

def delete_violation(incident_id):
    db_path = os.path.join("events", "violations.db")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM violations WHERE incident_id = ?", (incident_id,))
        conn.commit()
        conn.close()
        st.toast("Record deleted.", icon="🗑️")
        time.sleep(1)
        st.rerun()
    except Exception as e:
        st.error(f"Error: {e}")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Settings")
    video_source = st.text_input("Camera/Video Source", value="0")
    if "process" not in st.session_state:
        st.session_state.process = None
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🚀 Run", use_container_width=True):
            if st.session_state.process is None:
                st.session_state.process = subprocess.Popen(["python", "app.py", "--source", video_source])
                st.success("System Live")
    with c2:
        if st.button("🛑 Stop", use_container_width=True):
            if st.session_state.process:
                st.session_state.process.terminate()
                st.session_state.process = None
                st.info("Stopped")

# --- DATA PROCESSING ---
df = fetch_data()

if not df.empty:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # الترتيب الحقيقي (من الأقدم للأحدث) لتعيين الأرقام التسلسلية
    df = df.sort_values('timestamp', ascending=True).reset_index(drop=True)
    df['No.'] = range(1, len(df) + 1)
    
    # حساب السوابق لكل مركبة بناءً على هذا الترتيب
    df['prev_count'] = df.groupby('plate_text').cumcount()

    # --- TOP HEADER ---
    head_col1, head_col2 = st.columns([3, 1])
    with head_col1:
        st.markdown('<div class="main-title">🚦 Traffic Enforcement Dashboard</div>', unsafe_allow_html=True)
    with head_col2:
        st.metric(label="Total Violations Logged", value=len(df))

    st.divider()

    # --- SEARCH & SELECTION ---
    st.subheader("🕵️ Inspector View")
    search_query = st.text_input("🔍 Search by Plate Number", placeholder="Type plate...")

    if search_query:
        view_df = df[df['plate_text'].str.contains(search_query, case=False, na=False)]
    else:
        view_df = df.copy()

    if view_df.empty:
        st.warning("No records found.")
        st.stop()

    # التعديل المطلوب: العرض يبدأ بآخر مخالفة تلقائياً
    total_records = len(view_df)
    col_sel, col_empty = st.columns([1, 2])
    with col_sel:
        selected_index = st.number_input(
            f"Select Record (1 to {total_records})",
            min_value=1,
            max_value=total_records,
            value=total_records  # جعل القيمة الافتراضية هي الأخيرة (الأحدث)
        )

    # استخراج البيانات المختارة (يتم طرح 1 لأن الباندا تبدأ من الصفر)
    target = view_df.iloc[selected_index - 1]
    
    # --- KPIs ---
    k1, k2 = st.columns(2)
    with k1:
        st.markdown(f'<div class="kpi-box"><span style="color:#64748b;">SELECTED VEHICLE</span><br><span class="kpi-value">{target["plate_text"]}</span></div>', unsafe_allow_html=True)
    with k2:
        history_val = target['prev_count']
        color = "#ef4444" if history_val > 0 else "#10b981"
        st.markdown(f'<div class="kpi-box" style="border-bottom: 4px solid {color};"><span style="color:#64748b;">{"REPEAT OFFENDER" if history_val > 0 else "FIRST TIME RECORD"}</span><br><span class="kpi-value" style="color:{color};">{history_val} Prior Violations</span></div>', unsafe_allow_html=True)

    # --- MAIN CARD ---
    st.markdown(f"""
        <div class="violation-card">
            <div class="plate-display">{target['plate_text']}</div>
            <div style="display: flex; gap: 30px; color: #64748b; font-size: 15px; margin-top:10px;">
                <div><b>Violation No:</b> #{target['No.']}</div>
                <div><b>Date:</b> {target['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</div>
                <div><b>AI Confidence:</b> {target['confidence']}</div>
                <div><b>Ref ID:</b> {target['incident_id']}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # --- IMAGES ---
    if target['vehicle_image_path'] and os.path.exists(target['vehicle_image_path']):
        st.image(target['vehicle_image_path'], caption="Full Vehicle View", use_container_width=True)

    c_img1, c_img2 = st.columns(2)
    with c_img1:
        if target['plate_crop_path'] and os.path.exists(target['plate_crop_path']):
            st.image(target['plate_crop_path'], caption="Plate Crop", use_container_width=True)
    with c_img2:
        if target['plate_enhanced_path'] and os.path.exists(target['plate_enhanced_path']):
            st.image(target['plate_enhanced_path'], caption="AI Enhanced", use_container_width=True)

    if st.button(f"🗑️ Delete Record #{target['No.']}", use_container_width=True):
        delete_violation(target['incident_id'])

    st.divider()

    # --- LOG TABLE (عرض الأحدث أولاً في الجدول للراحة البصرية) ---
    st.subheader("📜 Historical Log")
    st.dataframe(
        df.sort_values('timestamp', ascending=False),
        use_container_width=True,
        hide_index=True,
        column_order=("No.", "plate_text", "timestamp", "confidence", "prev_count")
    )

else:
    st.info("System Ready. Database is empty.")

time.sleep(12)
st.rerun()

