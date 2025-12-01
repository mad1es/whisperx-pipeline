import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import json
import warnings
from scipy import stats
from pydub import AudioSegment
import io
import base64

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="multimodal speech analysis",
    layout="wide"
)


@st.cache_data
def load_data(data_path: str) -> pd.DataFrame:
    if Path(data_path).exists():
        df = pd.read_csv(data_path)
        
        if 'label' in df.columns:
            if 'label_x' in df.columns:
                df = df.drop(columns=['label_x'])
            if 'label_y' in df.columns:
                df = df.drop(columns=['label_y'])
        elif 'label_x' in df.columns and 'label_y' in df.columns:
            df['label'] = df['label_x'].fillna(df['label_y'])
            df = df.drop(columns=['label_x', 'label_y'])
        elif 'label_x' in df.columns:
            df['label'] = df['label_x']
            df = df.drop(columns=['label_x'])
        elif 'label_y' in df.columns:
            df['label'] = df['label_y']
            df = df.drop(columns=['label_y'])
        
        return df
    return pd.DataFrame()


@st.cache_data
def load_transcript(transcript_path: str) -> dict:
    if Path(transcript_path).exists():
        with open(transcript_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def extract_audio_segment(audio_path: str, start_time: float, end_time: float) -> bytes:
    try:
        audio = AudioSegment.from_wav(audio_path)
        start_ms = int(start_time * 1000)
        end_ms = int(end_time * 1000)
        segment = audio[start_ms:end_ms]
        
        buffer = io.BytesIO()
        segment.export(buffer, format="wav")
        buffer.seek(0)
        return buffer.read()
    except Exception as e:
        st.error(f"error extracting audio: {e}")
        return None


def plot_feature_distribution(df: pd.DataFrame, feature: str, group_col: str = 'label'):
    if feature not in df.columns or group_col not in df.columns:
        return None
    
    df_clean = df[[feature, group_col]].copy()
    df_clean = df_clean.dropna(subset=[feature, group_col])
    
    df_clean[group_col] = pd.to_numeric(df_clean[group_col], errors='coerce')
    df_clean = df_clean.dropna(subset=[group_col])
    
    df_clean[feature] = pd.to_numeric(df_clean[feature], errors='coerce')
    df_clean = df_clean.dropna(subset=[feature])
    
    if len(df_clean) == 0:
        return None
    
    unique_labels = sorted([l for l in df_clean[group_col].unique() if pd.notna(l)])
    if len(unique_labels) == 0:
        return None
    
    fig = go.Figure()
    
    for label in unique_labels:
        label_name = 'suicidal' if label == 1 else 'control'
        data = df_clean[df_clean[group_col] == label][feature]
        
        data = data.dropna()
        if len(data) > 0 and not data.isna().all():
            data_sorted = sorted(data.tolist())
            fig.add_trace(go.Violin(
                y=data_sorted,
                name=label_name,
                box_visible=True,
                meanline_visible=True,
                fillcolor='rgba(255,0,0,0.3)' if label == 1 else 'rgba(0,0,255,0.3)',
                line_color='red' if label == 1 else 'blue'
            ))
    
    if len(fig.data) == 0:
        return None
    
    fig.update_layout(
        title=f'feature distribution: {feature}',
        yaxis_title=feature,
        xaxis_title='group',
        height=500,
        showlegend=True
    )
    
    return fig


def plot_comparison_statistics(df: pd.DataFrame):
    if 'label' not in df.columns:
        return None
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['file_id', 'segment_id', 'label', 'start', 'end', 'duration']
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    if len(feature_cols) == 0:
        return None
    
    comparison_stats = []
    
    for col in feature_cols[:50]:
        group0 = df[df['label'] == 0][col].dropna()
        group1 = df[df['label'] == 1][col].dropna()
        
        if len(group0) > 0 and len(group1) > 0:
            try:
                stat, p_value = stats.mannwhitneyu(group0, group1, alternative='two-sided')
                effect_size = (group1.mean() - group0.mean()) / np.sqrt((group0.std()**2 + group1.std()**2) / 2) if (group0.std()**2 + group1.std()**2) > 0 else 0
                
                comparison_stats.append({
                    'feature': col,
                    'control_mean': group0.mean(),
                    'suicidal_mean': group1.mean(),
                    'difference': group1.mean() - group0.mean(),
                    'p_value': p_value,
                    'effect_size': effect_size,
                    'significant': p_value < 0.05
                })
            except:
                continue
    
    if not comparison_stats:
        return None
    
    stats_df = pd.DataFrame(comparison_stats).sort_values('p_value')
    
    fig = go.Figure()
    
    significant = stats_df[stats_df['significant']]
    non_significant = stats_df[~stats_df['significant']]
    
    if len(significant) > 0:
        fig.add_trace(go.Scatter(
            x=significant['difference'],
            y=-np.log10(significant['p_value'] + 1e-10),
            mode='markers+text',
            text=significant['feature'],
            textposition='top center',
            marker=dict(size=10, color='red', symbol='circle'),
            name='significant differences (p<0.05)',
            hovertemplate='<b>%{text}</b><br>difference: %{x:.4f}<br>-log10(p): %{y:.2f}<extra></extra>'
        ))
    
    if len(non_significant) > 0:
        fig.add_trace(go.Scatter(
            x=non_significant['difference'],
            y=-np.log10(non_significant['p_value'] + 1e-10),
            mode='markers',
            marker=dict(size=8, color='gray', symbol='circle', opacity=0.5),
            name='non-significant differences',
            hovertemplate='<b>%{text}</b><br>difference: %{x:.4f}<br>-log10(p): %{y:.2f}<extra></extra>',
            text=non_significant['feature']
        ))
    
    fig.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="red", 
                  annotation_text="p=0.05", annotation_position="right")
    
    fig.update_layout(
        title='volcano plot: group differences',
        xaxis_title='mean difference (suicidal - control)',
        yaxis_title='-log10(p-value)',
        height=600,
        hovermode='closest'
    )
    
    return fig, stats_df


def plot_interactive_timeline_multi(df_all: pd.DataFrame, selected_videos: list, transcripts_dict: dict):
    if len(selected_videos) == 0 or len(df_all) == 0:
        return None, None
    
    colors = px.colors.qualitative.Set3[:len(selected_videos)]
    color_map = {vid: colors[i % len(colors)] for i, vid in enumerate(selected_videos)}
    
    all_segments = []
    
    for video_id in selected_videos:
        df_video = df_all[df_all['file_id'] == video_id].sort_values('start')
        if len(df_video) == 0:
            continue
        
        for idx, row in df_video.iterrows():
            start = row.get('start', 0)
            end = row.get('end', 0)
            text = row.get('text', '')
            segment_id = row.get('segment_id', '')
            
            conf = row.get('asr_conf_mean', 0.5)
            
            pitch_col = [c for c in df_video.columns if 'F0semitoneFrom27.5Hz_sma3nz_amean' in c]
            pitch = row[pitch_col[0]] if pitch_col and pd.notna(row.get(pitch_col[0], np.nan)) else np.nan
            
            energy_col = [c for c in df_video.columns if 'loudness_sma3_amean' in c]
            energy = row[energy_col[0]] if energy_col and pd.notna(row.get(energy_col[0], np.nan)) else np.nan
            
            jitter_col = [c for c in df_video.columns if 'jitterLocal_sma3nz_amean' in c]
            jitter = row[jitter_col[0]] if jitter_col and pd.notna(row.get(jitter_col[0], np.nan)) else np.nan
            
            shimmer_col = [c for c in df_video.columns if 'shimmerLocaldB_sma3nz_amean' in c]
            shimmer = row[shimmer_col[0]] if shimmer_col and pd.notna(row.get(shimmer_col[0], np.nan)) else np.nan
            
            hnr_col = [c for c in df_video.columns if 'HNRdBACF_sma3nz_amean' in c]
            hnr = row[hnr_col[0]] if hnr_col and pd.notna(row.get(hnr_col[0], np.nan)) else np.nan
            
            all_segments.append({
                'video_id': video_id,
                'time': (start + end) / 2,
                'start': start,
                'end': end,
                'text': text,
                'segment_id': segment_id,
                'confidence': conf,
                'pitch': pitch,
                'energy': energy,
                'jitter': jitter,
                'shimmer': shimmer,
                'hnr': hnr
            })
    
    if len(all_segments) == 0:
        return None, None
    
    fig = make_subplots(
        rows=7, cols=1,
        subplot_titles=('asr confidence', 'pitch (f0)', 'energy (loudness)', 
                       'jitter (pitch instability)', 'shimmer (amplitude instability)', 
                       'hnr (harmonics/noise)', 'segment text'),
        vertical_spacing=0.08,
        row_heights=[0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.28],
        specs=[[{"secondary_y": False}]] * 7
    )
    
    for video_id in selected_videos:
        video_segments = [s for s in all_segments if s['video_id'] == video_id]
        if len(video_segments) == 0:
            continue
        
        color = color_map[video_id]
        times = [s['time'] for s in video_segments]
        starts = [s['start'] for s in video_segments]
        ends = [s['end'] for s in video_segments]
        texts = [s['text'] for s in video_segments]
        segment_ids = [s['segment_id'] for s in video_segments]
        
        confidences = [s['confidence'] for s in video_segments]
        conf_customdata = [(s['start'], s['end'], s['text'], s['segment_id'], s['video_id']) for s in video_segments]
        fig.add_trace(
            go.Scatter(
                x=times, y=confidences,
                mode='lines+markers',
                name=f'{video_id} - Confidence',
                line=dict(color=color, width=2),
                marker=dict(size=6, color=color),
                customdata=conf_customdata,
                hovertemplate='<b>%{customdata[4]}</b><br>confidence: %{y:.3f}<br>time: %{customdata[0]:.2f}-%{customdata[1]:.2f} sec<br>text: %{customdata[2]}<extra></extra>',
                legendgroup=video_id, showlegend=True
            ),
            row=1, col=1
        )
        
        pitches = [s['pitch'] for s in video_segments if pd.notna(s['pitch'])]
        pitch_times = [s['time'] for s in video_segments if pd.notna(s['pitch'])]
        pitch_customdata = [(s['start'], s['end'], s['text'], s['segment_id'], s['video_id']) for s in video_segments if pd.notna(s['pitch'])]
        if len(pitches) > 0:
            fig.add_trace(
                go.Scatter(
                    x=pitch_times, y=pitches,
                    mode='lines+markers',
                    name=f'{video_id} - Pitch',
                    line=dict(color=color, width=2),
                    marker=dict(size=6, color=color),
                    customdata=pitch_customdata,
                    hovertemplate='<b>%{customdata[4]}</b><br>pitch: %{y:.2f}<br>time: %{customdata[0]:.2f}-%{customdata[1]:.2f} sec<br>text: %{customdata[2]}<extra></extra>',
                    legendgroup=video_id, showlegend=False
                ),
                row=2, col=1
            )
        
        energies = [s['energy'] for s in video_segments if pd.notna(s['energy'])]
        energy_times = [s['time'] for s in video_segments if pd.notna(s['energy'])]
        energy_customdata = [(s['start'], s['end'], s['text'], s['segment_id'], s['video_id']) for s in video_segments if pd.notna(s['energy'])]
        if len(energies) > 0:
            fig.add_trace(
                go.Scatter(
                    x=energy_times, y=energies,
                    mode='lines+markers',
                    name=f'{video_id} - Energy',
                    line=dict(color=color, width=2),
                    marker=dict(size=6, color=color),
                    customdata=energy_customdata,
                    hovertemplate='<b>%{customdata[4]}</b><br>energy: %{y:.3f}<br>time: %{customdata[0]:.2f}-%{customdata[1]:.2f} sec<br>text: %{customdata[2]}<extra></extra>',
                    legendgroup=video_id, showlegend=False
                ),
                row=3, col=1
            )
        
        jitters = [s['jitter'] for s in video_segments if pd.notna(s['jitter'])]
        jitter_times = [s['time'] for s in video_segments if pd.notna(s['jitter'])]
        jitter_customdata = [(s['start'], s['end'], s['text'], s['segment_id'], s['video_id']) for s in video_segments if pd.notna(s['jitter'])]
        if len(jitters) > 0:
            fig.add_trace(
                go.Scatter(
                    x=jitter_times, y=jitters,
                    mode='lines+markers',
                    name=f'{video_id} - Jitter',
                    line=dict(color=color, width=2),
                    marker=dict(size=6, color=color),
                    customdata=jitter_customdata,
                    hovertemplate='<b>%{customdata[4]}</b><br>jitter: %{y:.4f}<br>time: %{customdata[0]:.2f}-%{customdata[1]:.2f} sec<br>text: %{customdata[2]}<extra></extra>',
                    legendgroup=video_id, showlegend=False
                ),
                row=4, col=1
            )
        
        shimmers = [s['shimmer'] for s in video_segments if pd.notna(s['shimmer'])]
        shimmer_times = [s['time'] for s in video_segments if pd.notna(s['shimmer'])]
        shimmer_customdata = [(s['start'], s['end'], s['text'], s['segment_id'], s['video_id']) for s in video_segments if pd.notna(s['shimmer'])]
        if len(shimmers) > 0:
            fig.add_trace(
                go.Scatter(
                    x=shimmer_times, y=shimmers,
                    mode='lines+markers',
                    name=f'{video_id} - Shimmer',
                    line=dict(color=color, width=2),
                    marker=dict(size=6, color=color),
                    customdata=shimmer_customdata,
                    hovertemplate='<b>%{customdata[4]}</b><br>shimmer: %{y:.4f}<br>time: %{customdata[0]:.2f}-%{customdata[1]:.2f} sec<br>text: %{customdata[2]}<extra></extra>',
                    legendgroup=video_id, showlegend=False
                ),
                row=5, col=1
            )
        
        hnrs = [s['hnr'] for s in video_segments if pd.notna(s['hnr'])]
        hnr_times = [s['time'] for s in video_segments if pd.notna(s['hnr'])]
        hnr_customdata = [(s['start'], s['end'], s['text'], s['segment_id'], s['video_id']) for s in video_segments if pd.notna(s['hnr'])]
        if len(hnrs) > 0:
            fig.add_trace(
                go.Scatter(
                    x=hnr_times, y=hnrs,
                    mode='lines+markers',
                    name=f'{video_id} - HNR',
                    line=dict(color=color, width=2),
                    marker=dict(size=6, color=color),
                    customdata=hnr_customdata,
                    hovertemplate='<b>%{customdata[4]}</b><br>hnr: %{y:.2f} db<br>time: %{customdata[0]:.2f}-%{customdata[1]:.2f} sec<br>text: %{customdata[2]}<extra></extra>',
                    legendgroup=video_id, showlegend=False
                ),
                row=6, col=1
            )
        
        for i, (start, end, text) in enumerate(zip(starts, ends, texts)):
            short_text = text[:50] + '...' if len(text) > 50 else text
            fig.add_trace(
                go.Scatter(
                    x=[start, end],
                    y=[i % 2, i % 2],
                    mode='lines+text',
                    text=[short_text],
                    textposition='middle center',
                    line=dict(width=2, color=color, dash='dot'),
                    showlegend=False,
                    customdata=[[start, end, text, segment_ids[i], video_id]],
                    hovertemplate=f'<b>{video_id}</b><br>{short_text}<br>time: {start:.2f}-{end:.2f} sec<extra></extra>',
                    legendgroup=video_id
                ),
                row=7, col=1
            )
    
    fig.update_xaxes(title_text="time (sec)", row=7, col=1)
    fig.update_yaxes(title_text="confidence", row=1, col=1)
    fig.update_yaxes(title_text="f0 (semitones)", row=2, col=1)
    fig.update_yaxes(title_text="energy", row=3, col=1)
    fig.update_yaxes(title_text="jitter", row=4, col=1)
    fig.update_yaxes(title_text="shimmer (db)", row=5, col=1)
    fig.update_yaxes(title_text="hnr (db)", row=6, col=1)
    
    fig.update_layout(
        height=1400,
        showlegend=True,
        hovermode='x unified',
        clickmode='event+select',
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=10, label="10s", step="second", stepmode="backward"),
                    dict(count=30, label="30s", step="second", stepmode="backward"),
                    dict(count=60, label="1min", step="second", stepmode="backward"),
                    dict(step="all", label="all")
                ])
            ),
            rangeslider=dict(visible=True, thickness=0.05),
            type="linear"
        ),
        dragmode='zoom'
    )
    
    for i in range(1, 8):
        fig.update_xaxes(
            rangeselector=None,
            rangeslider=None if i < 7 else dict(visible=True, thickness=0.05),
            type="linear"
        )
    
    return fig, all_segments


def plot_class_comparison_boxplots(df: pd.DataFrame, features: list):
    if 'label' not in df.columns or len(features) == 0:
        return None
    
    n_features = min(len(features), 12)
    features = features[:n_features]
    
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=features,
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    for idx, feature in enumerate(features):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        group0 = df[df['label'] == 0][feature].dropna()
        group1 = df[df['label'] == 1][feature].dropna()
        
        if len(group0) > 0 and len(group1) > 0:
            fig.add_trace(
                go.Box(y=group0, name='control', marker_color='blue', showlegend=(idx == 0)),
                row=row, col=col
            )
            fig.add_trace(
                go.Box(y=group1, name='suicidal', marker_color='red', showlegend=(idx == 0)),
                row=row, col=col
            )
    
    fig.update_layout(
        height=300 * n_rows, 
        showlegend=True,
        dragmode='zoom'
    )
    return fig


def main():
    st.title("multimodal speech analysis")
    st.markdown("---")
    
    data_path = st.sidebar.text_input(
        "data path",
        value="data/features/merged_features.csv"
    )
    
    df = load_data(data_path)
    
    if df.empty:
        st.warning("data not found. check file path.")
        st.info("""
        to start:
        1. run data processing pipeline
        2. ensure merged_features.csv exists
        """)
        return
    
    if 'label' in df.columns:
        df['label'] = pd.to_numeric(df['label'], errors='coerce')
        valid_labels = df['label'].dropna()
        if len(valid_labels) == 0:
            st.error("no valid labels in data. check merged_features.csv and metadata.csv")
            st.info("""
            solution:
            1. ensure data/metadata.csv has all videos with labels (0 or 1)
            2. run feature merging:
               python pipeline/merge_features.py --segments-metadata data/segments/segments_metadata.csv --opensmile-features data/features/opensmile_features.csv --output data/features/merged_features.csv --metadata data/metadata.csv --language ru
            """)
            return
    else:
        st.error("label column missing in data. check merged_features.csv")
        st.info("""
        solution:
        1. ensure data/metadata.csv exists with id and label columns
        2. run feature merging with metadata
        """)
        return
    
    if len(df) == 0:
        st.warning("no data after filtering. adjust filters.")
        return
    
    st.sidebar.markdown("### filters")
    
    if 'label' in df.columns:
        available_labels = sorted([float(l) for l in df['label'].dropna().unique() if pd.notna(l)])
        
        if len(available_labels) > 0:
            selected_labels = st.sidebar.multiselect(
                "groups",
                options=available_labels,
                default=available_labels,
                format_func=lambda x: 'suicidal' if x == 1.0 or x == 1 else 'control'
            )
            if len(selected_labels) > 0:
                df = df[df['label'].isin(selected_labels)]
            else:
                st.warning("select at least one group")
                return
        else:
            st.warning("no labels available. check metadata.csv")
            return
    
    if 'asr_conf_mean' in df.columns:
        min_conf = float(df['asr_conf_mean'].min())
        max_conf = float(df['asr_conf_mean'].max())
        conf_range = st.sidebar.slider(
            "ASR Confidence",
            min_value=min_conf,
            max_value=max_conf,
            value=(min_conf, max_conf)
        )
        df = df[
            (df['asr_conf_mean'] >= conf_range[0]) &
            (df['asr_conf_mean'] <= conf_range[1])
        ]
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "overview", "distributions", "timeline",
        "correlations", "class comparison", "statistics"
    ])
    
    with tab1:
        st.header("overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("total segments", len(df))
        
        with col2:
            if 'file_id' in df.columns:
                st.metric("unique videos", df['file_id'].nunique())
        
        with col3:
            if 'label' in df.columns:
                suicidal_count = ((df['label'] == 1) | (df['label'] == 1.0)).sum()
                control_count = ((df['label'] == 0) | (df['label'] == 0.0)).sum()
                st.metric("suicidal segments", int(suicidal_count))
                st.metric("control segments", int(control_count))
        
        with col4:
            if 'duration' in df.columns:
                avg_duration = df['duration'].mean()
                st.metric("avg duration", f"{avg_duration:.2f} sec")
        
        if 'label' in df.columns:
            unique_labels = df['label'].dropna().unique()
            if len(unique_labels) >= 2:
                st.subheader("group comparison")
                
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                exclude_cols = ['file_id', 'segment_id', 'label', 'start', 'end', 'duration']
                feature_cols = [col for col in numeric_cols if col not in exclude_cols]
                
                comparison_data = []
                for col in feature_cols[:30]:
                    group0 = df[(df['label'] == 0) | (df['label'] == 0.0)][col].dropna()
                    group1 = df[(df['label'] == 1) | (df['label'] == 1.0)][col].dropna()
                    
                    if len(group0) > 0 and len(group1) > 0:
                        try:
                            stat, p_value = stats.mannwhitneyu(group0, group1, alternative='two-sided')
                            comparison_data.append({
                                'feature': col,
                                'control mean': f"{group0.mean():.4f}",
                                'suicidal mean': f"{group1.mean():.4f}",
                                'difference': f"{group1.mean() - group0.mean():.4f}",
                                'p-value': f"{p_value:.6f}",
                                'significant': 'yes' if p_value < 0.05 else 'no'
                            })
                        except:
                            continue
                
                if comparison_data:
                    comparison_df = pd.DataFrame(comparison_data).sort_values('p-value')
                    st.dataframe(comparison_df.head(20), use_container_width=True)
    
    with tab2:
        st.header("feature distributions")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['file_id', 'segment_id', 'label', 'start', 'end', 'duration']
        feature_options = [col for col in numeric_cols if col not in exclude_cols]
        
        if feature_options and 'label' in df.columns:
            selected_feature = st.selectbox("select feature", feature_options)
            
            if selected_feature:
                fig = plot_feature_distribution(df, selected_feature)
                if fig:
                    st.plotly_chart(
                        fig, 
                        use_container_width=True,
                        config={
                            'displayModeBar': True,
                            'modeBarButtonsToAdd': ['zoom2d', 'pan2d', 'select2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d'],
                            'displaylogo': False
                        }
                    )
                    
                    group0 = df[df['label'] == 0][selected_feature].dropna()
                    group1 = df[df['label'] == 1][selected_feature].dropna()
                    
                    if len(group0) > 0 and len(group1) > 0:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("control mean", f"{group0.mean():.4f}")
                        with col2:
                            st.metric("suicidal mean", f"{group1.mean():.4f}")
                        with col3:
                            diff = group1.mean() - group0.mean()
                            st.metric("difference", f"{diff:.4f}")
        else:
            st.info("no features available or label missing")
    
    with tab3:
        st.header("interactive timeline")
        st.markdown("**select one or more videos to compare. click on graph point to play audio segment**")
        
        if 'file_id' in df.columns:
            video_ids = sorted(df['file_id'].unique())
            selected_videos = st.multiselect(
                "select videos",
                options=video_ids,
                default=video_ids[:min(3, len(video_ids))] if len(video_ids) > 0 else []
            )
            
            if len(selected_videos) == 0:
                st.warning("select at least one video")
            else:
                transcripts_dict = {}
                for video_id in selected_videos:
                    transcript_path = f"data/transcripts/{video_id}.json"
                    transcripts_dict[video_id] = load_transcript(transcript_path)
                
                fig, all_segments = plot_interactive_timeline_multi(df, selected_videos, transcripts_dict)
                
                if fig:
                    selected_points = st.plotly_chart(
                        fig, 
                        use_container_width=True, 
                        key="timeline_chart_multi",
                        on_select="rerun",
                        config={
                            'displayModeBar': True,
                            'modeBarButtonsToAdd': ['zoom2d', 'pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d'],
                            'displaylogo': False,
                            'toImageButtonOptions': {
                                'format': 'png',
                                'filename': 'timeline_chart',
                                'height': 1400,
                                'width': 1200,
                                'scale': 1
                            }
                        }
                    )
                    
                    if selected_points and 'selection' in selected_points:
                        selection = selected_points['selection']
                        if selection and 'points' in selection and len(selection['points']) > 0:
                            point = selection['points'][0]
                            if 'customdata' in point:
                                start_time, end_time, text, segment_id, video_id = point['customdata']
                                
                                st.subheader("selected segment")
                                col1, col2 = st.columns([2, 1])
                                with col1:
                                    st.markdown(f"**video:** {video_id}")
                                    st.markdown(f"**text:** {text}")
                                    st.markdown(f"**time:** {start_time:.2f} - {end_time:.2f} sec")
                                    st.markdown(f"**duration:** {end_time - start_time:.2f} sec")
                                
                                audio_path = f"data/audio_wav/{video_id}.wav"
                                if Path(audio_path).exists():
                                    audio_bytes = extract_audio_segment(audio_path, start_time, end_time)
                                    if audio_bytes:
                                        st.audio(audio_bytes, format='audio/wav')
                                else:
                                    st.warning(f"audio file not found: {audio_path}")
                    
                    st.subheader("or select segment from list")
                    if all_segments:
                        segment_options = [
                            f"{s['video_id']} | {s['start']:.2f}-{s['end']:.2f} sec: {s['text'][:50]}..."
                            for s in all_segments
                        ]
                        selected_segment_idx = st.selectbox(
                            "segment", 
                            range(len(segment_options)), 
                            format_func=lambda x: segment_options[x]
                        )
                        
                        if selected_segment_idx < len(all_segments):
                            seg = all_segments[selected_segment_idx]
                            
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                st.markdown(f"**video:** {seg['video_id']}")
                                st.markdown(f"**text:** {seg['text']}")
                                st.markdown(f"**time:** {seg['start']:.2f} - {seg['end']:.2f} sec")
                                st.markdown(f"**duration:** {seg['end'] - seg['start']:.2f} sec")
                                
                                if pd.notna(seg['pitch']):
                                    st.metric("pitch (f0)", f"{seg['pitch']:.2f}")
                                if pd.notna(seg['energy']):
                                    st.metric("energy", f"{seg['energy']:.3f}")
                                if pd.notna(seg['jitter']):
                                    st.metric("jitter", f"{seg['jitter']:.4f}")
                                if pd.notna(seg['shimmer']):
                                    st.metric("shimmer", f"{seg['shimmer']:.4f}")
                                if pd.notna(seg['hnr']):
                                    st.metric("hnr", f"{seg['hnr']:.2f} db")
                            
                            audio_path = f"data/audio_wav/{seg['video_id']}.wav"
                            if Path(audio_path).exists():
                                audio_bytes = extract_audio_segment(audio_path, seg['start'], seg['end'])
                                if audio_bytes:
                                    st.audio(audio_bytes, format='audio/wav')
                            else:
                                st.warning(f"audio file not found: {audio_path}")
        else:
            st.info("no video file information")
    
    with tab4:
        st.header("correlation matrix")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['file_id', 'segment_id', 'label', 'start', 'end', 'duration']
        feature_options = [col for col in numeric_cols if col not in exclude_cols]
        
        if len(feature_options) > 1:
            selected_features = st.multiselect(
                "select features",
                options=feature_options,
                default=feature_options[:15] if len(feature_options) >= 15 else feature_options
            )
            
            if len(selected_features) > 1:
                fig = plot_correlation_heatmap(df, selected_features)
                st.plotly_chart(
                    fig, 
                    use_container_width=True,
                    config={
                        'displayModeBar': True,
                        'modeBarButtonsToAdd': ['zoom2d', 'pan2d', 'select2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d'],
                        'displaylogo': False
                    }
                )
            else:
                st.info("select at least 2 features")
        else:
            st.info("not enough features for correlation analysis")
    
    with tab5:
        st.header("class comparison")
        
        if 'label' not in df.columns or df['label'].nunique() < 2:
            st.warning("data with labels for both classes (0 and 1) required")
            return
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['file_id', 'segment_id', 'label', 'start', 'end', 'duration']
        feature_options = [col for col in numeric_cols if col not in exclude_cols]
        
        st.subheader("top 12 features with largest differences")
        
        if len(feature_options) > 0:
            comparison_stats = []
            for col in feature_options:
                group0 = df[df['label'] == 0][col].dropna()
                group1 = df[df['label'] == 1][col].dropna()
                
                if len(group0) > 0 and len(group1) > 0:
                    try:
                        stat, p_value = stats.mannwhitneyu(group0, group1, alternative='two-sided')
                        effect_size = abs((group1.mean() - group0.mean()) / np.sqrt((group0.std()**2 + group1.std()**2) / 2)) if (group0.std()**2 + group1.std()**2) > 0 else 0
                        comparison_stats.append({
                            'feature': col,
                            'p_value': p_value,
                            'effect_size': effect_size,
                            'difference': abs(group1.mean() - group0.mean())
                        })
                    except:
                        continue
            
            if comparison_stats:
                top_features = pd.DataFrame(comparison_stats).nsmallest(12, 'p_value')['feature'].tolist()
                
                fig = plot_class_comparison_boxplots(df, top_features)
                if fig:
                    st.plotly_chart(
                        fig, 
                        use_container_width=True,
                        config={
                            'displayModeBar': True,
                            'modeBarButtonsToAdd': ['zoom2d', 'pan2d', 'select2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d'],
                            'displaylogo': False
                        }
                    )
                
                st.subheader("interpretation")
                st.markdown("""
                **what to look for in features:**
                - **pitch (f0)** - pitch: monotony may indicate depression
                - **jitter/shimmer** - voice instability: increased values may indicate stress/tension
                - **hnr** - harmonics to noise ratio: low values = hoarseness, tension
                - **energy/loudness** - volume: reduced energy may indicate apathy
                - **speech rate** - speech speed: slowed speech may be a sign of depression
                - **pause duration** - pause length: increased pauses may indicate difficulties
                """)
            else:
                st.info("failed to compute difference statistics")
        else:
            st.info("no features available for comparison")
    
    with tab6:
        st.header("difference statistics")
        
        if 'label' not in df.columns or df['label'].nunique() < 2:
            st.warning("data with labels for both classes required")
            return
        
        fig, stats_df = plot_comparison_statistics(df)
        
        if fig:
            st.plotly_chart(
                fig, 
                use_container_width=True,
                config={
                    'displayModeBar': True,
                    'modeBarButtonsToAdd': ['zoom2d', 'pan2d', 'select2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d'],
                    'displaylogo': False
                }
            )
            
            st.subheader("top 20 most significant differences")
            st.dataframe(
                stats_df.head(20)[['feature', 'control_mean', 'suicidal_mean', 'difference', 'p_value', 'effect_size', 'significant']],
                use_container_width=True
            )
            
            st.markdown("""
            **how to read the plot:**
            - **x-axis**: difference in means (positive = higher in suicidal, negative = lower)
            - **y-axis**: statistical significance (-log10(p-value))
            - **red points**: statistically significant differences (p < 0.05)
            - **gray points**: non-significant differences
            
            **effect size** shows effect magnitude:
            - < 0.2: small effect
            - 0.2-0.5: medium effect  
            - > 0.5: large effect
            """)


def plot_correlation_heatmap(df: pd.DataFrame, features: list = None):
    if features is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['file_id', 'segment_id', 'label', 'start', 'end', 'duration']
        features = [col for col in numeric_cols if col not in exclude_cols][:20]
    
    corr_matrix = df[features].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 8}
    ))
    
    fig.update_layout(
        title='feature correlation matrix',
        height=700,
        xaxis_title='features',
        yaxis_title='features',
        dragmode='zoom'
    )
    
    return fig


if __name__ == '__main__':
    main()
