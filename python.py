import streamlit as st
import json
import pandas as pd
import numpy_financial as npf
from math import ceil
import numpy as np
import time

# --- Cấu hình API và Model ---
# Lưu ý: __API_KEY__ sẽ được Canvas tự động cung cấp trong môi trường chạy.
# Nếu bạn muốn chạy cục bộ, hãy thay thế bằng khóa API của bạn.
API_KEY = ""
API_URL_GEMINI = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"
MODEL_NAME = "gemini-2.5-flash-preview-05-20" 

# --- Hàm gọi API Gemini với Cơ chế Backoff ---
async def call_gemini_api_with_retry(payload, max_retries=5):
    """Gọi API Gemini với cơ chế thử lại (exponential backoff)."""
    headers = {'Content-Type': 'application/json'}
    
    # Kiểm tra và thêm khóa API nếu cần thiết
    url = f"{API_URL_GEMINI}?key={API_KEY}" if API_KEY else API_URL_GEMINI

    for attempt in range(max_retries):
        try:
            # Giả lập Fetch API trong môi trường này
            response = await st.runtime.scriptrunner.add_script_run_ctx(
                fetch(url, method='POST', headers=headers, body=json.dumps(payload))
            )
            
            result = await response.json()
            
            if response.status != 200:
                # Nếu lỗi do server hoặc rate limit, thử lại
                error_message = result.get('error', {}).get('message', 'Lỗi không xác định từ API')
                if response.status in [429, 500, 503]:
                    st.toast(f"Lỗi API (HTTP {response.status}): Thử lại lần {attempt + 1}. Chi tiết: {error_message}")
                    if attempt < max_retries - 1:
                        await time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                raise Exception(f"Lỗi API (HTTP {response.status}): {error_message}")

            # Xử lý kết quả trả về
            candidate = result.get('candidates', [{}])[0]
            if not candidate:
                return None

            # Đối với responseMimeType: "application/json", nội dung nằm trong parts[0].text
            json_content = candidate['content']['parts'][0]['text']
            return json_content

        except Exception as e:
            if attempt == max_retries - 1:
                st.error(f"Thao tác thất bại sau {max_retries} lần thử. Lỗi: {e}")
                raise
            # Nếu là lỗi mạng hoặc lỗi khác, chờ và thử lại
            await time.sleep(2 ** attempt)
    return None

# --- Nhiệm vụ 1: AI Trích xuất Dữ liệu Cấu trúc ---
def get_extraction_payload(document_text):
    """Tạo payload cho Gemini để trích xuất dữ liệu tài chính."""
    
    # Định nghĩa Schema JSON bắt buộc
    response_schema = {
        "type": "OBJECT",
        "properties": {
            "Vốn đầu tư": { "type": "NUMBER", "description": "Tổng vốn đầu tư ban đầu, đơn vị Tỷ VNĐ" },
            "Dòng đời dự án": { "type": "INTEGER", "description": "Số năm hoạt động của dự án" },
            "Doanh thu/năm": { "type": "NUMBER", "description": "Doanh thu hàng năm, đơn vị Tỷ VNĐ" },
            "Chi phí/năm": { "type": "NUMBER", "description": "Chi phí hoạt động hàng năm, đơn vị Tỷ VNĐ" },
            "WACC (%)": { "type": "NUMBER", "description": "Tỷ lệ WACC của doanh nghiệp, tính bằng phần trăm" },
            "Thuế suất (%)": { "type": "NUMBER", "description": "Tỷ lệ thuế TNDN, tính bằng phần trăm" }
        },
        "required": ["Vốn đầu tư", "Dòng đời dự án", "Doanh thu/năm", "Chi phí/năm", "WACC (%)", "Thuế suất (%)"]
    }
    
    system_prompt = (
        "Bạn là một bộ máy trích xuất dữ liệu tài chính. Hãy đọc kỹ văn bản được cung cấp và trích xuất sáu (6) thông tin tài chính bắt buộc sau: "
        "Vốn đầu tư (chuyển sang tỷ VNĐ nếu cần), Dòng đời dự án (số năm), Doanh thu/năm (tỷ VNĐ), Chi phí/năm (tỷ VNĐ), WACC (%), và Thuế suất (%). "
        "Trả lời bằng một đối tượng JSON TUYỆT ĐỐI tuân thủ schema đã cung cấp. Đơn vị tiền tệ (tỷ VNĐ) không được bao gồm trong giá trị."
    )

    payload = {
        "contents": [{ "parts": [{ "text": document_text }] }],
        "systemInstruction": { "parts": [{ "text": system_prompt }] },
        "config": {
            "responseMimeType": "application/json",
            "responseSchema": response_schema
        }
    }
    return payload

async def extract_data_from_document(document_text):
    """Thực hiện trích xuất dữ liệu bằng AI."""
    # st.session_state.extraction_loading = True # Được xử lý bởi nút bấm và st.rerun
    payload = get_extraction_payload(document_text)
    
    try:
        response_json_string = await call_gemini_api_with_retry(payload)
        
        # Xử lý chuỗi JSON và chuyển thành dictionary
        if response_json_string:
            # Loại bỏ các ký tự Markdown không cần thiết (nếu AI trả về ```json...```)
            if response_json_string.startswith("```json"):
                response_json_string = response_json_string.strip().replace("```json\n", "").replace("\n```", "")
                
            data = json.loads(response_json_string)
            st.session_state.project_data = data
            st.session_state.cash_flow_calculated = False
            st.success("Trích xuất dữ liệu thành công!")
        else:
            st.error("Không thể trích xuất dữ liệu. Vui lòng kiểm tra lại nội dung đầu vào.")

    except json.JSONDecodeError:
        st.error("Lỗi giải mã JSON từ AI. Vui lòng thử lại hoặc điều chỉnh văn bản.")
        # st.info(f"Phản hồi thô từ AI: {response_json_string}")
    except Exception as e:
        st.error(f"Lỗi trong quá trình gọi API: {e}")
    
    st.session_state.extraction_loading = False
    
# --- Nhiệm vụ 2 & 3: Xây dựng Dòng tiền & Tính toán Chỉ số ---
def calculate_project_metrics(data):
    """Tính toán dòng tiền và các chỉ số NPV, IRR, PP, DPP."""
    
    # Chuyển đổi dữ liệu sang dạng số và tỷ lệ
    try:
        # Chuyển đổi Tỷ VNĐ sang VNĐ
        investment = data['Vốn đầu tư'] * 1_000_000_000
        lifespan = int(data['Dòng đời dự án'])
        revenue = data['Doanh thu/năm'] * 1_000_000_000
        cost = data['Chi phí/năm'] * 1_000_000_000
        tax_rate = data['Thuế suất (%)'] / 100
        wacc = data['WACC (%)'] / 100
    except KeyError as e:
        st.error(f"Thiếu thông tin quan trọng để tính toán: {e}")
        return None, None
    except TypeError as e:
        st.error(f"Dữ liệu trích xuất không đúng định dạng số: {e}")
        return None, None
    
    # 1. Tính Dòng tiền Thuần Hàng năm (NCF - Net Cash Flow)
    EBIT = revenue - cost  # Lợi nhuận trước lãi vay và thuế (Giả định lãi vay = 0 để tính dòng tiền dự án)
    Tax_Amount = EBIT * tax_rate
    NPAT = EBIT - Tax_Amount  # Lợi nhuận thuần sau thuế (Net Profit After Tax)
    # Giả định NCF = NPAT + Khấu hao (Khấu hao = 0 để đơn giản)
    NCF_Annual = NPAT
    
    # 2. Xây dựng Bảng Dòng tiền (Cash Flow Schedule)
    years = list(range(lifespan + 1))
    cash_flows = [-investment] + [NCF_Annual] * lifespan
    
    df_cash_flow = pd.DataFrame({
        'Năm': years,
        # Hiển thị tất cả dưới dạng Tỷ VNĐ để dễ đọc
        'Dòng Tiền (- Đầu tư)': np.array(cash_flows) / 1e9,
        'Lợi Nhuận Gộp (Tỷ VNĐ)': [0.0] + [EBIT / 1e9] * lifespan,
        'NCF Hàng Năm (Tỷ VNĐ)': np.array([-investment / 1e9] + [NCF_Annual / 1e9] * lifespan)
    }).set_index('Năm')

    # 3. Tính toán các Chỉ số
    cash_flows_array = np.array(cash_flows)
    
    # a. NPV (Net Present Value)
    npv_value = npf.npv(wacc, cash_flows_array[1:]) + cash_flows_array[0] # npf.npv không bao gồm CF0
    
    # b. IRR (Internal Rate of Return)
    try:
        irr_value = npf.irr(cash_flows_array)
        if np.isnan(irr_value):
            irr_value = float('nan')
    except:
        irr_value = float('nan') # Không thể tính nếu dòng tiền không đổi dấu

    # c. PP (Payback Period - Thời gian hoàn vốn)
    cumulative_cf = np.cumsum(cash_flows_array)
    pp_year_index = np.where(cumulative_cf >= 0)[0]
    
    if pp_year_index.size > 0:
        pp_full_year = pp_year_index[0]
        if pp_full_year > 0:
            previous_cf = cumulative_cf[pp_full_year - 1] 
            cf_of_payback_year = cash_flows_array[pp_full_year]
            pp_value = pp_full_year - 1 + abs(previous_cf) / cf_of_payback_year if cf_of_payback_year != 0 else float('inf')
        else:
            pp_value = 0
    else:
        pp_value = float('inf') # Không bao giờ hoàn vốn
        
    # d. DPP (Discounted Payback Period - Thời gian hoàn vốn có chiết khấu)
    discount_factors = 1 / (1 + wacc)**np.array(years)
    discounted_cf = cash_flows_array * discount_factors
    cumulative_dcf = np.cumsum(discounted_cf)
    
    dpp_year_index = np.where(cumulative_dcf >= 0)[0]

    if dpp_year_index.size > 0:
        dpp_full_year = dpp_year_index[0]
        if dpp_full_year > 0:
            previous_dcf = cumulative_dcf[dpp_full_year - 1]
            dcf_of_payback_year = discounted_cf[dpp_full_year]
            dpp_value = dpp_full_year - 1 + abs(previous_dcf) / dcf_of_payback_year if dcf_of_payback_year != 0 else float('inf')
        else:
            dpp_value = 0
    else:
        dpp_value = float('inf')
        
    
    results = {
        "NPV": npv_value,
        "IRR": irr_value,
        "PP": pp_value,
        "DPP": dpp_value
    }
    
    return df_cash_flow, results

# --- Nhiệm vụ 4: AI Phân tích Chỉ số ---
async def analyze_project_metrics(analysis_results, project_data):
    """Yêu cầu AI phân tích các chỉ số tài chính đã tính toán."""
    # st.session_state.analysis_loading = True # Được xử lý bởi nút bấm và st.rerun
    
    # Xử lý IRR NaN
    irr_display = f"{analysis_results['IRR']:.2%}" if not np.isnan(analysis_results['IRR']) else "Không thể tính (Dòng tiền không đổi dấu)"

    analysis_prompt = f"""
    Bạn là một chuyên gia phân tích tài chính đầu tư. Hãy phân tích tính khả thi của dự án dựa trên các thông số sau và đưa ra kết luận:
    
    1. THÔNG SỐ DỰ ÁN ĐÃ TRÍCH XUẤT (Tỷ VNĐ, %):
       - Vốn đầu tư: {project_data.get('Vốn đầu tư')} Tỷ VNĐ
       - Dòng đời: {project_data.get('Dòng đời dự án')} năm
       - Doanh thu/năm: {project_data.get('Doanh thu/năm')} Tỷ VNĐ
       - Chi phí/năm: {project_data.get('Chi phí/năm')} Tỷ VNĐ
       - WACC: {project_data.get('WACC (%)')}%
       - Thuế suất: {project_data.get('Thuế suất (%)')}%
       
    2. KẾT QUẢ ĐÁNH GIÁ HIỆU QUẢ DỰ ÁN (VNĐ, %):
       - NPV (Giá trị hiện tại ròng): {analysis_results['NPV']:,.0f} VNĐ
       - IRR (Tỷ suất sinh lời nội bộ): {irr_display}
       - PP (Thời gian hoàn vốn): {analysis_results['PP']:.2f} năm
       - DPP (Thời gian hoàn vốn có chiết khấu): {analysis_results['DPP']:.2f} năm
       
    Yêu cầu phân tích:
    - Đánh giá NPV: Dự án có đáng giá không? (NPV > 0 là chấp nhận)
    - Đánh giá IRR: So sánh IRR với WACC (13.00%). Nếu IRR < WACC thì không khả thi.
    - Đánh giá PP và DPP: Phân tích thời gian hoàn vốn có nhanh không, có chấp nhận được không.
    - Kết luận tổng thể và đưa ra khuyến nghị (Ví dụ: Cần tăng doanh thu hay giảm chi phí).
    
    Viết phân tích chi tiết, chuyên nghiệp bằng Tiếng Việt.
    """
    
    payload = {
        "contents": [{ "parts": [{ "text": analysis_prompt }] }],
        "systemInstruction": { "parts": [{ "text": "Bạn là chuyên gia phân tích tài chính. Hãy cung cấp một bản phân tích chuyên sâu về hiệu quả dự án." }] },
    }
    
    try:
        analysis_text = await call_gemini_api_with_retry(payload)
        st.session_state.ai_analysis = analysis_text
        st.success("Phân tích AI hoàn tất!")
    except Exception as e:
        st.session_state.ai_analysis = f"Lỗi phân tích từ AI: {e}"
        st.error("Không thể hoàn thành phân tích. Vui lòng kiểm tra kết nối API.")
        
    st.session_state.analysis_loading = False

# --- Cấu hình Streamlit App ---
st.set_page_config(layout="wide", page_title="Đánh giá Dự án Đầu tư bằng AI")

# --- Khởi tạo Session State ---
if 'project_data' not in st.session_state:
    st.session_state.project_data = {}
if 'cash_flow_data' not in st.session_state:
    st.session_state.cash_flow_data = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'ai_analysis' not in st.session_state:
    st.session_state.ai_analysis = ""
if 'extraction_loading' not in st.session_state:
    st.session_state.extraction_loading = False
if 'analysis_loading' not in st.session_state:
    st.session_state.analysis_loading = False

# Dữ liệu mẫu (nội dung file Word mô phỏng từ yêu cầu)
SAMPLE_DOCUMENT_TEXT = """
Đây là phương án kinh doanh chi tiết cho dự án đầu tư dây chuyền sản xuất bánh mì:
Tổng vốn đầu tư ban đầu cần thiết cho dự án là 30 tỷ VNĐ. Dự án có vòng đời hoạt động dự kiến là 10 năm.
Kể từ cuối năm thứ 1, dự án dự kiến tạo ra doanh thu ổn định hàng năm là 3,5 tỷ VNĐ.
Chi phí hoạt động hàng năm ước tính là 2 tỷ VNĐ (chưa bao gồm thuế).
WACC của doanh nghiệp (chi phí vốn) được xác định là 13%.
Thuế suất Thu nhập Doanh nghiệp áp dụng là 20%.
"""

# --- Giao diện Ứng dụng ---
st.title("🥖 AI Phân Tích Hiệu Quả Dự Án Kinh Doanh")
st.markdown("Sử dụng Gemini để trích xuất dữ liệu, tính toán tài chính và phân tích tính khả thi của dự án.")

# Sử dụng tab để tổ chức luồng công việc
tab1, tab2, tab3 = st.tabs(["1. Trích Xuất Dữ Liệu (AI)", "2. Bảng Dòng Tiền & Chỉ Số", "3. Phân Tích Kết Quả (AI)"])

with tab1:
    st.header("1. Tải Lên & Trích Xuất Thông Tin Dự Án")
    
    st.info("Vì môi trường chạy không hỗ trợ đọc file Word, bạn vui lòng dán hoặc sử dụng **Văn bản mẫu** để mô phỏng dữ liệu đầu vào.")
    
    document_text = st.text_area(
        "Dán nội dung Phương án Kinh doanh (hoặc nội dung file Word):",
        value=SAMPLE_DOCUMENT_TEXT,
        height=300
    )
    
    col_button, col_status = st.columns([1, 3])
    with col_button:
        if st.button("Trích Xuất Dữ Liệu Bằng AI", type="primary", disabled=st.session_state.extraction_loading):
            if document_text:
                # Reset states
                st.session_state.ai_analysis = "" 
                st.session_state.project_data = {} 
                st.session_state.cash_flow_data = None 
                st.session_state.analysis_results = {} 
                st.session_state.extraction_loading = True
                st.toast("Đang gọi AI để trích xuất dữ liệu...")
                st.rerun() # Trigger rerun to show loading state
                
            else:
                st.warning("Vui lòng dán nội dung dự án vào ô văn bản.")
    
    # Logic chạy hàm async sau khi bấm nút (nếu state loading đang Bật)
    if st.session_state.extraction_loading:
        with st.spinner('AI đang đọc và trích xuất dữ liệu tài chính...'):
             # Chạy hàm async
             st.runtime.scriptrunner.add_script_run_ctx(
                 extract_data_from_document(document_text)
             )
    
    if st.session_state.project_data:
        st.subheader("Kết Quả Dữ Liệu Đã Trích Xuất")
        
        # Chuyển đổi sang DataFrame để hiển thị đẹp hơn
        data_display = pd.DataFrame(st.session_state.project_data.items(), 
                                    columns=['Chỉ Tiêu', 'Giá Trị'])
        
        data_display['Đơn Vị'] = data_display['Chỉ Tiêu'].apply(
            lambda x: 'Tỷ VNĐ' if 'Tỷ' in x or 'đầu tư' in x or 'Doanh thu' in x or 'Chi phí' in x else '%' if '%' in x else 'Năm'
        )
        data_display['Giá Trị'] = data_display.apply(
            lambda row: f"{row['Giá Trị']:,.2f}" if row['Đơn Vị'] == 'Tỷ VNĐ' else f"{row['Giá Trị']:,.0f}" if row['Đơn Vị'] == 'Năm' else f"{row['Giá Trị']:.2f}", axis=1
        )
        data_display['Giá Trị'] = data_display['Giá Trị'] + ' ' + data_display['Đơn Vị']
        data_display = data_display[['Chỉ Tiêu', 'Giá Trị']]

        st.dataframe(data_display, hide_index=True)
        
        # Tự động tính toán khi có dữ liệu
        if not st.session_state.cash_flow_data:
            df_cf, results = calculate_project_metrics(st.session_state.project_data)
            st.session_state.cash_flow_data = df_cf
            st.session_state.analysis_results = results
            st.session_state.cash_flow_calculated = True

with tab2:
    st.header("2. Bảng Dòng Tiền & Các Chỉ Số Tài Chính")
    
    if st.session_state.cash_flow_data is not None:
        
        st.subheader("Bảng Dòng Tiền Thuần (NCF)")
        
        # Hiển thị NPV, IRR, PP, DPP
        st.markdown("---")
        st.subheader("Các Chỉ Số Hiệu Quả Đầu Tư")
        
        results = st.session_state.analysis_results
        
        col_npv, col_irr, col_pp, col_dpp = st.columns(4)

        col_npv.metric(
            label="NPV (Giá trị hiện tại ròng)", 
            value=f"{results['NPV']:,.0f} VNĐ", 
            delta_color="off"
        )
        
        irr_display = f"{results['IRR']:.2%}" if not np.isnan(results['IRR']) else "N/A"
        col_irr.metric(
            label="IRR (Tỷ suất sinh lời nội bộ)", 
            value=irr_display,
            delta_color="off"
        )
        
        # Format PP và DPP
        def format_payback(value):
            if value == float('inf'):
                return "Không hoàn vốn"
            return f"{value:.2f} năm"
            
        col_pp.metric(
            label="PP (Thời gian hoàn vốn)", 
            value=format_payback(results['PP']),
            delta_color="off"
        )
        col_dpp.metric(
            label="DPP (Thời gian hoàn vốn có chiết khấu)", 
            value=format_payback(results['DPP']),
            delta_color="off"
        )
        
        st.markdown("---")
        st.caption("Chi tiết Dòng tiền (Đơn vị: Tỷ VNĐ)")
        
        st.dataframe(
            st.session_state.cash_flow_data.style.format({
                'Dòng Tiền (- Đầu tư)': '{:,.2f}',
                'Lợi Nhuận Gộp (Tỷ VNĐ)': '{:,.2f}',
                'NCF Hàng Năm (Tỷ VNĐ)': '{:,.2f}'
            }),
            use_container_width=True
        )
        
    else:
        st.warning("Vui lòng trích xuất dữ liệu ở Tab 1 trước.")

with tab3:
    st.header("3. Phân Tích Chuyên Sâu Của AI")
    
    if st.session_state.cash_flow_data is not None:
        if st.button("Yêu Cầu AI Phân Tích Chỉ Số", type="primary", disabled=st.session_state.analysis_loading):
            st.session_state.analysis_loading = True
            st.toast("Đang gọi AI để phân tích...")
            st.rerun() # Trigger rerun to show loading state
            
        if st.session_state.analysis_loading:
            with st.spinner('AI đang tổng hợp và phân tích dữ liệu...'):
                 # Chạy hàm async
                 st.runtime.scriptrunner.add_script_run_ctx(
                     analyze_project_metrics(st.session_state.analysis_results, st.session_state.project_data)
                 )

        if st.session_state.ai_analysis:
            st.subheader("Báo Cáo Đánh Giá Hiệu Quả Đầu Tư")
            st.markdown(st.session_state.ai_analysis)
            st.info("Phân tích này giúp Chủ đầu tư có cái nhìn tổng quan về tính khả thi của dự án.")
        elif not st.session_state.analysis_loading:
            st.info("Nhấn nút trên để AI bắt đầu phân tích kết quả tính toán.")
    else:
        st.warning("Vui lòng hoàn thành bước Trích xuất Dữ liệu và Tính toán Chỉ số trước.")
