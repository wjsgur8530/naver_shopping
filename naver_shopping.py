import os
import json
import traceback
import requests
import streamlit as st
from dotenv import load_dotenv

# .env에서 환경 변수 불러오기
load_dotenv()
HCX_API_KEY = os.getenv("HCX_API_KEY")
MODEL_NAME = "HCX-DASH-002"
URL = f"https://clovastudio.stream.ntruss.com/testapp/v3/chat-completions/{MODEL_NAME}"

NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")

def sendInputAndFunctionDefinition(messages, temperature=0.3, top_p=0.8, max_tokens=1024, top_k=0, repeat_penalty=1.1):
    """HCX 모델에 요청을 보내는 함수 (항상 tools 포함)"""
    headers = {
        "Authorization": f"Bearer {HCX_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "messages": messages,
        "seed": 0,
        "topP": top_p,
        "topK": top_k,
        "maxTokens": max_tokens,
        "temperature": temperature,
        "repeatPenalty": repeat_penalty,
        "stopBefore": []
    }

    # tools 정의 추가 (항상 추가)
    data["tools"] = [
        {
            "type": "function",
            "function": {
                "name": "get_shopping",
                "description": "쇼핑, 상품 정보를 알려주는 도구",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "검색어. 예: 나이키 신발"
                        },
                        "display": {
                            "type": "integer",
                            "description": "한 번에 표시할 검색 결과 개수 (최대 100)"
                        },
                        "start": {
                            "type": "integer",
                            "description": "검색 시작 위치 (최대 100)"
                        },
                        "sort": {
                            "type": "string",
                            "description": "정렬 방법: sim, date, asc, dsc"
                        },
                        "filter": {
                            "type": "string",
                            "description": "상품 유형 필터: naverpay 등"
                        },
                        "exclude": {
                            "type": "string",
                            "description": "제외할 유형: used, rental, cbshop"
                        },
                    },
                    "required": ["query"]
                }
            }
        }
    ]
    data["toolChoice"] = "auto"  # 모델이 자동으로 도구 사용 여부 결정

    response = requests.post(URL, headers=headers, json=data)
    return response.json()

# 네이버 쇼핑 API
def get_search_naver_shopping(query, display=None, start=None, sort=None, filter=None, exclude=None):
    """네이버 쇼핑 검색 API를 호출하는 함수"""
    api_url = "https://openapi.naver.com/v1/search/shop.json"
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
    }
    params = {"query": query}
    if display is not None:
        params["display"] = display
    if start is not None:
        params["start"] = start
    if sort is not None:
        params["sort"] = sort
    if filter is not None:
        params["filter"] = filter
    if exclude is not None:
        params["exclude"] = exclude

    response = requests.get(api_url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.status_code, "message": response.text}

def invokeFunctionFromResponse(function_to_call, user_prompt, display=None, start=None, sort=None):
    """모델 응답을 분석하고 적절한 함수 호출 또는 일반 텍스트 응답을 처리"""

    result = function_to_call.get("result")
    message = result.get("message") if result else None
    content = message.get("content") if message else None
    tool_calls = message.get("toolCalls") if message else None
    try:
        if tool_calls:
            try:
                tool_call = tool_calls[0]
                function_name = tool_call["function"]["name"]
                arguments = tool_call["function"]["arguments"]

                if isinstance(arguments, str):
                    arguments = json.loads(arguments)

                if function_name == "get_shopping":
                    # 지원하는 함수인 경우 도구 실행
                    shopping_result = get_search_naver_shopping(user_prompt, display, start, sort)

                    # API 응답을 받은 후 확장 기능 및 추가적인 액션 가능
                    return {
                        "shopping_result": shopping_result,
                        "used_tool": True
                    }
                else:
                    # 정의되지 않은 함수 호출
                    return {
                        "error": f"지원하지 않는 함수입니다: {function_name}",
                        "used_tool": True
                    }

            except (KeyError, IndexError, json.JSONDecodeError) as e:
                return {
                    "error": f"toolCalls 파싱 오류: {str(e)}",
                    "traceback": traceback.format_exc(),
                    "used_tool": False
                }

        else:
            return {
                "content": content,
                "query": message,
                "used_tool": False
            }

    except Exception as e:
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "used_tool": False
        }

def sendFunctionResult(messages, temperature=0.3, top_p=0.8, max_tokens=1024, top_k=0, repeat_penalty=1.1):
    """API 결과로 최종 답변 생성"""
    headers = {
        "Authorization": f"Bearer {HCX_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "messages": messages,
        "seed": 0,
        "topP": top_p,
        "topK": top_k,
        "maxTokens": max_tokens,
        "temperature": temperature,
        "repeatPenalty": repeat_penalty,
        "stopBefore": []
    }

    response = requests.post(URL, headers=headers, json=data)
    response.raise_for_status()
    return response.json()

def safe_json_dumps(obj):
    def default(o):
        try:
            return str(o)
        except Exception:
            return "<non-serializable>"
    return json.dumps(obj, ensure_ascii=False, default=default)


# Streamlit UI 설정
st.set_page_config(page_title="HyperCLOVA X 멀티모달 쇼핑 Assistant", layout="wide")
st.title("🛍️ HyperCLOVA X + 멀티모달 쇼핑 도우미")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 대화 기록 표시
with st.container():
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.write(chat["content"])

# 입력 필드
prompt = st.chat_input(placeholder="쇼핑 상품을 입력해주세요.")

# 사이드바 설정
with st.sidebar:
    st.header("⚙️ 모델 설정")
    temperature = st.slider("🌡️ Temperature", 0.0, 1.0, 0.3, 0.1)
    top_p = st.slider("🔢 Top P", 0.0, 1.0, 0.8, 0.05)
    top_k = st.number_input("🔢 Top K", min_value=0, max_value=100, value=0)
    max_tokens = st.slider("✍️ Max Tokens", 64, 4096, 1024, 64)
    repeat_penalty = st.slider("🔁 Repeat Penalty", 0.5, 2.0, 1.1, 0.1)

    st.header("📊 네이버 API 조절")
    display = st.slider("display", 0, 100, 3, 1)
    start = st.slider("start", 0, 100, 1, 1)
    sort = st.selectbox("sort", ("sim", "asc", "date"))

if prompt:
    # 사용자 메시지를 대화 기록에 추가
    user_message = {"role": "user", "content": prompt}
    st.session_state.chat_history.append(user_message)
    st.session_state.messages.append(user_message)

    with st.chat_message("user"):
        st.write(prompt)

    # 이후 로직 계속 실행 (함수 호출, 응답 생성 등)
    with st.spinner("🧠 HCX 모델이 생각 중..."):
        try:
            # 1단계: 모델에 요청 보내기 (항상 tools 포함)
            messages = st.session_state.messages.copy()

            # 프롬프트 설정
            messages.append({
                "role": "system",
                "content": "당신은 똑똑한 네이버 쇼핑 도우미입니다. 네이버 API에서 가져온 응답을 깔끔하게 정리해서 사용자에게 전달하는 것이 주요 목표입니다."
            })

            # 답변 설정
            messages.append({
                "role": "assistant",
                "content": "최종 응답은 네이버 쇼핑 데이터를 기준으로 상품명, 총평으로 정리해야 합니다. (예시) 나이키 신발을 추천해달라는 사용자 입력이 있었을 때 정확하게 신발을 추천해줘야 합니다. (신발을 검색했는데 신발 끈을 추천해주는 일은 없어야 합니다.)"
            })


            st.write("messages:")
            st.write(messages)
            function_to_call = sendInputAndFunctionDefinition(messages, temperature, top_p, max_tokens, top_k, repeat_penalty)
            st.write("function_to_call:")
            st.write(function_to_call)

            # toolCalls 존재 여부 확인
            result = function_to_call.get("result", None)
            message = result.get("message") if result else None
            tool_calls = message.get("toolCalls") if message else None

            if tool_calls:
                # 2단계: 응답 처리 및 쇼핑 API 호출
                function_result = invokeFunctionFromResponse(function_to_call, prompt, display, start, sort)
                st.write("function_result:")
                st.write(function_result)

                shopping_result = function_result.get("shopping_result", {})
                st.write("shopping_result:")
                st.write(shopping_result)
                used_tool = function_result.get("used_tool", False)
                st.write("used_tool:")
                st.write(used_tool)

                # 3단계: 쇼핑 결과 표시
                items = shopping_result.get("items", [])
                if items:
                    cols_per_row = 3
                    for i in range(0, len(items), cols_per_row):
                        cols = st.columns(cols_per_row)
                        for col, item in zip(cols, items[i:i + cols_per_row]):
                            with col:
                                st.image(item["image"], width=150)
                                st.markdown(f"**{item['title']}**", unsafe_allow_html=True)
                                st.markdown(f"💸 가격: **{item['lprice']}원**")
                                st.markdown(f"🏬 쇼핑몰: {item['mallName']}")
                                st.markdown(f"[🔗 상품 보기]({item['link']})")

                # tool 사용된 메시지를 추가
                tool_name = tool_calls[0]["function"]["name"]
                messages.append({
                    "role": "tool",
                    "toolCallId": tool_calls[0]["id"],
                    "content": str(function_result),
                })
            else:
                used_tool = False  # tool 사용되지 않음

            # 4단계: 모델 응답 표시
            with st.chat_message("assistant"):
                if tool_calls:
                    st.markdown(f"<div style='background-color: #f0f7ff; padding: 5px; border-radius: 3px; display: inline-block; margin-bottom: 10px;'>🛠️ ToolCalls - {tool_name} 사용됨</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div style='background-color: #f5f5f5; padding: 5px; border-radius: 3px; display: inline-block; margin-bottom: 10px;'>⚠️ ToolCalls 사용되지 않음</div>", unsafe_allow_html=True)

                # 5단계: 최종 응답 생성 및 표시
                final_result = sendFunctionResult(messages, temperature, top_p, max_tokens, top_k, repeat_penalty)
                st.write(final_result)
                final_result_content = final_result.get("result", {}).get("message", {}).get("content", "검색 결과입니다.")
                st.write(final_result_content)


            # 응답을 대화 기록에 추가
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": final_result_content
            })

            # 메시지 히스토리 업데이트
            if used_tool and tool_calls and len(tool_calls) > 0:
                try:
                    # 도구를 사용한 경우 - 메시지 히스토리에 도구 호출 및 응답 추가
                    st.session_state.messages.append(function_to_call["result"]["message"])
                    st.session_state.messages.append({
                        "role": "tool",
                        "toolCallId": tool_calls[0]["id"],
                        "content": safe_json_dumps(shopping_result)
                    })
                    # 최종 응답 추가
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": final_result_content
                    })
                except (KeyError, IndexError) as e:
                    st.error(f"toolCalls 처리 중 오류 발생: {e}")
                    st.json(tool_calls)
            else:
                # 일반 응답인 경우
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_result_content
                })

        except Exception as e:
            st.error(f"❌ 오류 발생: {e}")
            st.code(traceback.format_exc())
