
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from scipy.stats import pearsonr
import math
print(st.__version__)
print(cv2.__version__)
print(mp.__version__)
# --- MEDIA PIPE UTILS ---
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import drawing_styles as mp_drawing_styles

# ==========================================
# 1. DATA STRUCTURES (From original design)
# ==========================================

@dataclass
class Landmark:
    x: float; y: float; z: float; visibility: float

@dataclass
class PoseData:
    landmarks: List[Landmark]
    world_landmarks: List[Landmark]
    timestamp: float
    frame_index: int
    raw_landmarks: any

@dataclass
class MetricData:
    knee_angle: float; trunk_lean: float; hip_shoulder_sync: float
    phase: str; heel_lift: bool; frame_index: int

@dataclass
class RepSummary:
    rep_index: int; start_frame: int; end_frame: int
    depth_score: float; trunk_score: float; heel_score: float
    sync_score: float; eccentric_tempo: float; concentric_tempo: float

@dataclass
class SetSummary:
    reps: List[RepSummary]; overall_score: float
    letter_grade: str; radar_scores: Dict[str, float]; raw_averages: Dict[str, float]

# ==========================================
# 2. CORE LOGIC MODULES (Refactored for Web)
# ==========================================

class PoseProcessor:
    def __init__(self):
        self.pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def process_video(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        all_pose_data = []
        frame_idx = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        progress_bar = st.progress(0)
        status_text = st.empty()

        while cap.isOpened():
            success, frame = cap.read()
            if not success: break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)

            if results.pose_landmarks and results.pose_world_landmarks:
                landmarks = [Landmark(l.x, l.y, l.z, l.visibility) for l in results.pose_landmarks.landmark]
                world_landmarks = [Landmark(l.x, l.y, l.z, l.visibility) for l in results.pose_world_landmarks.landmark]
                all_pose_data.append(PoseData(landmarks, world_landmarks, frame_idx/fps, frame_idx, results.pose_landmarks))
            
            frame_idx += 1
            if frame_idx % 10 == 0:
                progress = frame_idx / total_frames
                progress_bar.progress(min(progress, 1.0))
                status_text.text(f"Processing Frame {frame_idx}/{total_frames}...")

        cap.release()
        status_text.text("Analysis Complete!")
        return all_pose_data, fps

class BiomechanicalAnalyzer:
    def __init__(self):
        self.L_HIP, self.L_KNEE, self.L_ANKLE, self.L_SHOULDER, self.L_HEEL = 23, 25, 27, 11, 29
        self.L_HEEL, self.L_FOOT_INDEX = 29, 31 

    def _calculate_angle(self, a, b, c):
        ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
        bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

    def analyze(self, pose_list, fps):
        metrics = []
        # Use initial frame to establish baseline height for heel lift detection
        initial_height = abs(pose_list[0].landmarks[self.L_HIP].y - pose_list[0].landmarks[self.L_ANKLE].y)
        
        for p in pose_list:
            w, l = p.world_landmarks, p.landmarks
            knee = self._calculate_angle(w[self.L_HIP], w[self.L_KNEE], w[self.L_ANKLE])
            
            # Trunk lean relative to vertical
            trunk_vec = np.array([w[self.L_SHOULDER].x - w[self.L_HIP].x, w[self.L_SHOULDER].y - w[self.L_HIP].y])
            trunk_lean = np.degrees(np.arccos(np.dot(trunk_vec, [0, -1]) / np.linalg.norm(trunk_vec)))
            
            # Heel lift logic
            #heel_lift = (pose_list[0].landmarks[self.L_HEEL].y - l[self.L_HEEL].y) > (initial_height * 0.05)
            heel_lift = l[self.L_HEEL].y < (l[self.L_FOOT_INDEX].y - 0.02)
            
            metrics.append(MetricData(knee, trunk_lean, 0.0, 'START', heel_lift, p.frame_index))

        # 1. State Machine for Phase Detection
        angles = pd.Series([m.knee_angle for m in metrics]).rolling(5, center=True).mean().fillna(method='bfill').values
        for i in range(1, len(metrics)):
            diff = angles[i] - angles[i-1]
            if angles[i] > 160: phase = "START"
            elif diff < -0.5: phase = "DESCENT"
            elif abs(diff) < 0.2 and metrics[i-1].phase == "DESCENT": phase = "BOTTOM"
            elif diff > 0.5: phase = "ASCENT"
            else: phase = metrics[i-1].phase
            metrics[i].phase = phase

        # 2. Repetition Segmentation & Sync Calculation
        reps, in_rep, start_idx = [], False, 0
        for i, m in enumerate(metrics):
            if m.phase == "DESCENT" and not in_rep: 
                in_rep, start_idx = True, i
            
            if m.phase == "START" and in_rep:
                rep_frames = pose_list[start_idx:i]
                rep_metrics = metrics[start_idx:i]
                
                if len(rep_frames) > 15:
                    # --- DYNAMIC SYNC CALCULATION ---
                    # We only look at the ASCENT phase for hip-shoulder sync
                    ascent_indices = [idx for idx, rm in enumerate(rep_metrics) if rm.phase == "ASCENT"]
                    
                    if len(ascent_indices) > 5:
                        # Extract vertical (Y) positions of Hip and Shoulder
                        hip_y = [rep_frames[idx].landmarks[self.L_HIP].y for idx in ascent_indices]
                        sho_y = [rep_frames[idx].landmarks[self.L_SHOULDER].y for idx in ascent_indices]
                        
                        # Calculate Velocity (difference between frames)
                        hip_v = np.diff(hip_y)
                        sho_v = np.diff(sho_y)
                        
                        # Correlation: 1.0 means they move perfectly together
                        # If hips rise (Y decreases) while shoulders lag, correlation drops
                        corr, _ = pearsonr(hip_v, sho_v)
                        sync_score = max(0, corr * 100) 
                    else:
                        sync_score = 50.0 # Default if ascent is too fast to measure
                    
                    reps.append(RepSummary(
                        rep_index=len(reps),
                        start_frame=start_idx,
                        end_frame=i,
                        depth_score=min([rm.knee_angle for rm in rep_metrics]),
                        trunk_score=max([rm.trunk_lean for rm in rep_metrics]),
                        heel_score=0.0 if any([rm.heel_lift for rm in rep_metrics]) else 100.0,
                        sync_score=sync_score,
                        eccentric_tempo=len([rm for rm in rep_metrics if rm.phase=="DESCENT"])/fps,
                        concentric_tempo=len([rm for rm in rep_metrics if rm.phase=="ASCENT"])/fps
                    ))
                in_rep = False
                
        return metrics, reps

class GradingEngine:
    def grade(self, reps):
        if not reps: return SetSummary([], 0, "F", {})
        
         # Raw Values
        raw = {
            'Depth': np.mean([r.depth_score for r in reps]),
            'Trunk': np.mean([r.trunk_score for r in reps]),
            'Ecc Tempo': np.mean([r.eccentric_tempo for r in reps]),
            'Conc Tempo': np.mean([r.concentric_tempo for r in reps]),
            'Sync': np.mean([r.sync_score for r in reps]),
            'Heel Lifts': sum([1 for r in reps if r.heel_score == 0])
        }

        # Calculate averages across all reps
        scores = {
            'Depth': np.mean([min(100, ((180-r.depth_score)/90)*100) for r in reps]),
            'Trunk': np.mean([max(0, 100 - ((r.trunk_score-20)/30*100)) for r in reps]),
            'Heel': np.mean([r.heel_score for r in reps]),
            'Sync': np.mean([r.sync_score for r in reps]), 
            'Ecc Tempo': np.mean([100 if 2 <= r.eccentric_tempo <= 4 else 100 * math.exp(-abs(r.eccentric_tempo-3)/1.5) for r in reps]),
            'Conc Tempo': np.mean([100 if r.concentric_tempo <= 3 else 100 * math.exp(-(r.concentric_tempo-3)/1.5) for r in reps])
        }
        
        weights = {
            'Depth': 0.30, 
            'Trunk': 0.25, 
            'Heel': 0.15, 
            'Sync': 0.15, 
            'Ecc Tempo': 0.10, 
            'Conc Tempo': 0.05
        }
        
        overall = sum(scores[k] * weights[k] for k in weights)
        
        # Letter Grade Mapping
        if overall >= 90: grade = 'A'
        elif overall >= 80: grade = 'B'
        elif overall >= 70: grade = 'C'
        elif overall >= 60: grade = 'D'
        else: grade = 'F'
        
        return SetSummary(reps, overall, grade, scores, raw)

# ==========================================
# 3. STREAMLIT UI
# ==========================================

st.set_page_config(page_title="Squat AI Pro", layout="wide")
st.title("Squat Form Analyzer")
st.sidebar.header("Upload Workout")
uploaded_file = st.sidebar.file_uploader("Choose a video file", type=['mp4', 'mov', 'avi'])

if uploaded_file is not None:
    # Save uploaded file to temp
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    col1, col2 = st.columns([1, 1])

    with st.spinner('Analyzing movement...'):
        # 1. Process
        processor = PoseProcessor()
        pose_data, fps = processor.process_video(tfile.name)
        
        # 2. Analyze
        analyzer = BiomechanicalAnalyzer()
        metrics, reps = analyzer.analyze(pose_data, fps)
        
        # 3. Grade
        grader = GradingEngine()
        summary = grader.grade(reps)

    with col1:
        st.subheader("Form Assessment")
        # Radar Chart
        categories = list(summary.radar_scores.keys())
        values = list(summary.radar_scores.values())
        values += values[:1]
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
        ax.fill(angles, values, color='teal', alpha=0.3)
        ax.plot(angles, values, color='teal', linewidth=2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        st.pyplot(fig)

        st.metric("Overall Score", f"{int(summary.overall_score)}%", summary.letter_grade)
        st.write(f"Total Reps Detected: {len(reps)}")

        st.divider()
        st.subheader("Coaching Insights")
        
        raw = summary.raw_averages
        
        # Displaying metrics in three columns for better layout
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.write(f"**Depth:** {int(raw['Depth'])} degrees")
            if raw['Depth'] > 100:
                st.warning("Increase depth. Target thighs parallel to the floor.")
            else:
                st.success("Optimal depth achieved.")

            st.write(f"**Trunk Lean:** {int(raw['Trunk'])} degrees")
            if raw['Trunk'] > 45:
                st.warning("Keep chest higher. Torso is leaning too far forward.")
            else:
                st.success("Torso angle is stable.")

        with c2:
            st.write(f"**Down Tempo:** {raw['Ecc Tempo']:.1f}s")
            if raw['Ecc Tempo'] < 2.0:
                st.error("Descent is too fast. Control the weight for 2-3 seconds.")
            elif raw['Ecc Tempo'] > 4.5:
                st.warning("Descent is very slow. This may cause premature fatigue.")
            else:
                st.success("Excellent descent control.")

            st.write(f"**Up Tempo:** {raw['Conc Tempo']:.1f}s")
            if raw['Conc Tempo'] > 3.0:
                st.warning("Ascent is slow. Work on explosive power coming out of the hole.")
            else:
                st.success("Good upward drive.")

        with c3:
            st.write(f"**Sync Score:** {int(raw['Sync'])}/100")
            if raw['Sync'] < 70:
                st.error("Hips are rising faster than shoulders. Drive both up simultaneously.")
            elif raw['Sync'] < 85:
                st.warning("Slight lag between hip and shoulder drive.")
            else:
                st.success("Perfect hip-shoulder synchronization.")

            st.write(f"**Heel Stability:** {int(raw['Heel Lifts'])} lifts detected")
            if raw['Heel Lifts'] > 0:
                st.error(f"Heel lift detected on {int(raw['Heel Lifts'])} repetitions.")
            else:
                st.success("Heels remained stable.")

    with col2:
        st.subheader("Visual Feedback")
        # Annotate video for web playback
        output_path = "annotated_video.mp4"
        cap = cv2.VideoCapture(tfile.name)
        fourcc = cv2.VideoWriter_fourcc(*'avc1') # H.264 for Web
        width, height = int(cap.get(3)), int(cap.get(4))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        pose_map = {p.frame_index: p for p in pose_data}
        metric_map = {m.frame_index: m for m in metrics}

        for f_idx in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            ret, frame = cap.read()
            if not ret: break
            
            if f_idx in pose_map:
                mp_drawing.draw_landmarks(frame, pose_map[f_idx].raw_landmarks, mp_pose.POSE_CONNECTIONS,
                                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            if f_idx in metric_map:
                m = metric_map[f_idx]
                cv2.putText(frame, f"Phase: {m.phase}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                if m.heel_lift: cv2.putText(frame, "HEEL LIFT!", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            out.write(frame)
        
        cap.release()
        out.release()
        
        # Display Video
        if os.path.exists(output_path):
            st.video(output_path)
            
    st.sidebar.success("Analysis Finished!")
else:
    st.info("Please upload a lateral-view video of your squats to begin.")
