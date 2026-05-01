import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import os

st.title("NBA 도플갱어 찾기")
name = st.text_input("이름을 입력하세요")

if st.button("결과 보기"):
    st.write(f"{name}님과 닮은 선수는...")
    # 이 부분이 터미널에 찍힙니다!
    print(f"로그: 사용자가 '{name}' 이름으로 버튼을 클릭함")

# --- 데이터 로드 및 전처리 (캐싱 적용) ---
@st.cache_data
def load_and_preprocess_data():
    print("📢 캐시가 없어서 데이터를 새로 로드하고 계산합니다!")
    possible_paths = ['data/2023-2024 NBA Player Stats - Regular.csv', '2023-2024 NBA Player Stats - Regular.csv']
    df = None
    for path in possible_paths:
        if os.path.exists(path):
            df = pd.read_csv(path, sep=';', encoding='latin-1')
            break
    if df is None:
        st.error("🏀 데이터를 찾을 수 없습니다! CSV 파일 위치를 확인하세요.")
        st.stop()
    
    df = df.drop_duplicates(subset=['Player'], keep='first')
    df['Name'] = df['Player']           
    df['3PM'] = df['3P']                
    df['DEF'] = df['STL'] + df['BLK']
    
    stats_cols = ['PTS', 'AST', 'TRB', 'DEF', '3PM']
    stats_df = df[stats_cols].fillna(0)
    
    means, stds = stats_df.mean(), stats_df.std()
    z_stats_df = (stats_df - means) / stds
    
    return df, stats_df, z_stats_df, means, stds

df, stats_df, z_stats_df, league_means, league_stds = load_and_preprocess_data()

# --- 세션 상태 초기화 ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'quiz_step' not in st.session_state:
    st.session_state.quiz_step = 0
if 'user_stats' not in st.session_state:
    st.session_state.user_stats = {k: league_means[k] for k in league_means.index}

# --- 첫 화면 및 로그인 ---
if not st.session_state.logged_in:
    st.title("🏀 NBA 도플갱어 테스트")
    st.markdown("### 🏀 과제 제출자 정보")
    st.markdown("- **성명**: 박지민\n- **학번**: 2022204077")
    st.divider()
    st.write("본인의 정보를 입력하여 로그인하세요.")
    
    input_id = st.text_input("학번을 입력하세요")
    input_name = st.text_input("이름을 입력하세요")
    
    if st.button("로그인 및 테스트 시작"):
        if input_id == "2022204077" and input_name == "박지민":
            st.session_state.logged_in = True
            st.session_state.student_name = input_name
            st.rerun()
        else:
            st.error("🏀 로그인 실패: 학번 또는 이름이 등록된 정보와 다릅니다.")

# --- 퀴즈 진행 ---
else:
    with st.sidebar:
        st.write(f"로그인: {st.session_state.student_name}님")
        if st.button("로그아웃"):
            st.session_state.clear()
            st.rerun()

    st.title("🏀 NBA 도플갱어 찾기")
    st.progress(st.session_state.quiz_step / 10)

    questions = [
        ("Q1. [속공] 2대1 찬스에서 당신의 선택은?", "직접 레이업 마무리", "동료에게 앨리웁 패스", 'PTS', 'AST', 2.0, 1.5),
        ("Q2. [수비] 상대 팀이 픽앤롤 공격을 시도합니다.", "핸들러를 끝까지 압박", "수비 로테이션 지시", 'DEF', 'AST', 0.5, 1.0),
        ("Q3. [전환] 우리 팀의 슛이 림을 빗나갔습니다.", "공격 리바운드 시도", "빠르게 백코트 가담", 'TRB', 'DEF', 1.5, 0.5),
        ("Q4. [아이솔레이션] 돌파 중 기습 더블팀을 만났습니다.", "틈을 뚫고 파울 유도", "슈터에게 킥아웃 패스", 'PTS', 'AST', 1.5, 1.5),
        ("Q5. [외곽] 수비수가 거리를 두며 슛을 유도합니다.", "주저 없이 3점슛 시도", "폭발적으로 골밑 돌파", '3PM', 'PTS', 1.0, 1.5),
        ("Q6. [오프볼] 동료가 공격을 세팅 중입니다.", "외곽에서 슛 준비", "동료를 위해 스크린", '3PM', 'TRB', 0.8, 1.0),
        ("Q7. [허슬] 루즈볼이 라인 밖으로 나가려 합니다.", "몸을 날려 공을 살려냄", "안전하게 잡아 타임아웃", 'DEF', 'AST', 0.5, 0.5),
        ("Q8. [경합] 골밑에서 치열한 자리싸움 중입니다.", "완벽한 박스아웃", "뛰어올라 리바운드 낚아챔", 'AST', 'TRB', 0.5, 2.0),
        ("Q9. [분위기] 팀 사기가 떨어진 상황입니다.", "강력한 덩크로 분위기 반전", "거친 수비로 흐름 차단", 'PTS', 'DEF', 2.0, 0.5),
        ("Q10. [클러치] 종료 5초 전 1점 차로 지고 있습니다.", "역전 3점슛 던짐", "확실한 2점 돌파 시도", '3PM', 'PTS', 1.0, 1.5)
    ]

    if st.session_state.quiz_step < 10:
        q, op1, op2, s1, s2, v1, v2 = questions[st.session_state.quiz_step]
        st.subheader(q)
        if st.button(op1):
            st.session_state.user_stats[s1] += v1
            st.session_state.quiz_step += 1
            st.rerun()
        if st.button(op2):
            st.session_state.user_stats[s2] += v2
            st.session_state.quiz_step += 1
            st.rerun()

    # --- 최종 결과 ---
    elif st.session_state.quiz_step == 10:
        user_raw = np.array([st.session_state.user_stats[c] for c in ['PTS', 'AST', 'TRB', 'DEF', '3PM']])
        user_z_score = ((user_raw - league_means.values) / league_stds.values).reshape(1, -1)

        similarities = cosine_similarity(user_z_score, z_stats_df.values)
        best_match_idx = np.argmax(similarities)
        best_player = df.iloc[best_match_idx]
        player_name = best_player['Name']
        
        st.subheader(f"당신의 NBA 도플갱어는 **{player_name}** 선수입니다!")
        st.write(f"(일치율: {similarities[0][best_match_idx]*100:.1f}% | 소속팀: {best_player['Tm']})")
        
        youtube_url = f"https://www.youtube.com/results?search_query={player_name.replace(' ','+')}+highlights"
        st.markdown(f'<a href="{youtube_url}" target="_blank"><button style="width:100%; background-color:#FF0000; color:white; padding:10px; border:none; border-radius:5px; cursor:pointer; font-size:16px; font-weight:bold; margin-bottom:20px;">🏀 {player_name} 하이라이트 영상 보기</button></a>', unsafe_allow_html=True)
        
        display_categories = ['PTS', 'AST', 'RB', 'DEF', '3PTS']
        data_categories = ['PTS', 'AST', 'TRB', 'DEF', '3PM']
        max_vals = stats_df.max().values
        p_vals = np.concatenate((best_player[data_categories].values / max_vals, [best_player[data_categories].values[0] / max_vals[0]]))
        u_vals = np.concatenate((user_raw / max_vals, [user_raw[0] / max_vals[0]]))
        angles = np.linspace(0, 2*np.pi, len(display_categories), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.plot(angles, u_vals, color='red', label='My Style', linewidth=2)
        ax.fill(angles, u_vals, color='red', alpha=0.25)
        ax.plot(angles, p_vals, color='blue', label=player_name, linewidth=2)
        ax.fill(angles, p_vals, color='blue', alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(display_categories)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        st.pyplot(fig)

        if st.button("다시 하기"):
            st.session_state.quiz_step = 0
            st.session_state.user_stats = {k: league_means[k] for k in league_means.index}
            st.rerun()
