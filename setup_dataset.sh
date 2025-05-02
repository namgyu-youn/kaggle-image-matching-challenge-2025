#!/bin/bash

# 이미지 매칭 챌린지 2025 - 데이터셋 설정 스크립트
# 이 스크립트는 Kaggle에서 데이터를 다운로드하고 정리합니다.

set -e  # 오류 발생 시 스크립트 종료

# 색상 정의
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # 색상 없음

# 함수 정의
print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 기본 설정 값
DATA_DIR="./dataset"
DOWNLOAD_DIR="./download"
KAGGLE_COMPETITION="image-matching-challenge-2025"

# 명령줄 인수 처리
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --download_dir)
            DOWNLOAD_DIR="$2"
            shift 2
            ;;
        --help)
            echo "이미지 매칭 챌린지 2025 - 데이터셋 설정 스크립트"
            echo "사용법: $0 [옵션]"
            echo "옵션:"
            echo "  --data_dir DIR       데이터셋을 저장할 디렉토리 [기본값: ./dataset]"
            echo "  --download_dir DIR   다운로드 파일을 저장할 임시 디렉토리 [기본값: ./download]"
            echo "  --help               이 도움말 메시지 표시"
            exit 0
            ;;
        *)
            print_error "알 수 없는 인수: $1"
            exit 1
            ;;
    esac
done

# 1. Kaggle CLI 확인
print_step "Kaggle CLI 확인 중..."
if ! command -v kaggle &> /dev/null; then
    print_error "Kaggle CLI를 찾을 수 없습니다. 설치가 필요합니다."
    echo "다음 명령어를 실행하여 설치하세요:"
    echo "pip install kaggle"
    exit 1
fi
print_success "Kaggle CLI가 설치되어 있습니다."

# 2. Kaggle 인증 확인
print_step "Kaggle 인증 확인 중..."
if [ ! -f ~/.kaggle/kaggle.json ]; then
    print_error "Kaggle API 인증 정보를 찾을 수 없습니다."
    echo "다음 단계를 따라 Kaggle API 인증을 설정하세요:"
    echo "1. Kaggle 계정으로 로그인하세요: https://www.kaggle.com/"
    echo "2. 프로필 > 계정 > API > 새 API 토큰 생성을 클릭하세요."
    echo "3. 다운로드된 kaggle.json 파일을 ~/.kaggle/ 디렉토리에 넣으세요."
    echo "4. 파일 권한을 설정하세요: chmod 600 ~/.kaggle/kaggle.json"
    exit 1
fi
print_success "Kaggle 인증이 설정되어 있습니다."

# 3. 다운로드 디렉토리 생성
print_step "다운로드 디렉토리 생성 중..."
mkdir -p "$DOWNLOAD_DIR"
print_success "다운로드 디렉토리가 생성되었습니다: $DOWNLOAD_DIR"

# 4. 데이터셋 다운로드
print_step "Kaggle에서 데이터셋 다운로드 중..."
kaggle competitions download -c "$KAGGLE_COMPETITION" -p "$DOWNLOAD_DIR"
print_success "데이터셋이 다운로드되었습니다."

# 5. 압축 해제
print_step "압축 파일 해제 중..."
# 모든 zip 파일 찾기
zip_files=$(find "$DOWNLOAD_DIR" -name "*.zip")

# 각 zip 파일 처리
for zip_file in $zip_files; do
    echo "압축 해제 중: $zip_file"
    unzip -o "$zip_file" -d "$DOWNLOAD_DIR/extracted"
done
print_success "모든 압축 파일이 해제되었습니다."

# 6. 데이터셋 디렉토리 구조 생성
print_step "데이터셋 디렉토리 구조 생성 중..."
mkdir -p "$DATA_DIR/train"
mkdir -p "$DATA_DIR/test"
print_success "데이터셋 디렉토리 구조가 생성되었습니다."

# 7. 데이터셋 구성 및 이름 변경
print_step "데이터셋 구성 및 이름 변경 중..."

# 데이터셋 디렉토리 구조를 확인합니다
if [ -d "$DOWNLOAD_DIR/extracted/train" ]; then
    train_dir="$DOWNLOAD_DIR/extracted/train"
    echo "학습 데이터 디렉토리 발견: $train_dir"

    # 학습 디렉토리 내용 확인
    dataset_count=1
    for dir in "$train_dir"/*; do
        if [ -d "$dir" ]; then
            dataset_name=$(basename "$dir")
            target_dir="$DATA_DIR/train/dataset$dataset_count"

            echo "학습 데이터셋 구성: $dataset_name -> dataset$dataset_count"
            mkdir -p "$target_dir"
            cp -r "$dir"/* "$target_dir"/

            # 데이터셋 카운트 증가
            ((dataset_count++))
        fi
    done

    # 개별 파일 처리 (루트 레벨에 있는 이미지)
    file_count=0
    for file in "$train_dir"/*.{png,jpg,jpeg}; do
        if [ -f "$file" ]; then
            if [ $file_count -eq 0 ]; then
                # 첫 번째 이미지 파일을 발견하면 새 데이터셋 디렉토리 생성
                target_dir="$DATA_DIR/train/dataset$dataset_count"
                mkdir -p "$target_dir"
                echo "추가 학습 데이터셋 생성: dataset$dataset_count (개별 파일용)"
                ((dataset_count++))
            fi

            cp "$file" "$target_dir"/
            ((file_count++))
        fi
    done
else
    print_warning "표준 학습 디렉토리 구조를 찾을 수 없습니다. 데이터셋 구조를 수동으로 확인하세요."

    # 대안적인 접근: 추출된 디렉토리에서 모든 하위 디렉토리 검색
    dataset_count=1
    for dir in "$DOWNLOAD_DIR/extracted"/*; do
        if [ -d "$dir" ] && [ "$(basename "$dir")" != "test" ]; then
            dataset_name=$(basename "$dir")
            target_dir="$DATA_DIR/train/dataset$dataset_count"

            echo "학습 데이터셋 구성: $dataset_name -> dataset$dataset_count"
            mkdir -p "$target_dir"
            cp -r "$dir"/* "$target_dir"/

            # 데이터셋 카운트 증가
            ((dataset_count++))
        fi
    done
fi

# 테스트 데이터 처리
if [ -d "$DOWNLOAD_DIR/extracted/test" ]; then
    test_dir="$DOWNLOAD_DIR/extracted/test"
    echo "테스트 데이터 디렉토리 발견: $test_dir"

    # 테스트 디렉토리 내용 확인
    dataset_count=1
    for dir in "$test_dir"/*; do
        if [ -d "$dir" ]; then
            dataset_name=$(basename "$dir")
            target_dir="$DATA_DIR/test/dataset$dataset_count"

            echo "테스트 데이터셋 구성: $dataset_name -> dataset$dataset_count"
            mkdir -p "$target_dir"
            cp -r "$dir"/* "$target_dir"/

            # 데이터셋 카운트 증가
            ((dataset_count++))
        fi
    done

    # 개별 파일 처리 (루트 레벨에 있는 이미지)
    file_count=0
    for file in "$test_dir"/*.{png,jpg,jpeg}; do
        if [ -f "$file" ]; then
            if [ $file_count -eq 0 ]; then
                # 첫 번째 이미지 파일을 발견하면 새 데이터셋 디렉토리 생성
                target_dir="$DATA_DIR/test/dataset$dataset_count"
                mkdir -p "$target_dir"
                echo "추가 테스트 데이터셋 생성: dataset$dataset_count (개별 파일용)"
                ((dataset_count++))
            fi

            cp "$file" "$target_dir"/
            ((file_count++))
        fi
    done
else
    print_warning "표준 테스트 디렉토리 구조를 찾을 수 없습니다. 데이터셋 구조를 수동으로 확인하세요."
fi

print_success "데이터셋 구성 및 이름 변경이 완료되었습니다."

# 8. 데이터셋 요약 생성
print_step "데이터셋 요약 생성 중..."

# 학습 데이터셋 수 계산
train_count=$(find "$DATA_DIR/train" -maxdepth 1 -type d | wc -l)
train_count=$((train_count - 1))  # 상위 디렉토리 제외

# 테스트 데이터셋 수 계산
test_count=$(find "$DATA_DIR/test" -maxdepth 1 -type d | wc -l)
test_count=$((test_count - 1))  # 상위 디렉토리 제외

# 총 이미지 수 계산
train_images=$(find "$DATA_DIR/train" -type f -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" | wc -l)
test_images=$(find "$DATA_DIR/test" -type f -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" | wc -l)
total_images=$((train_images + test_images))

# 데이터셋 요약 파일 생성
cat > "$DATA_DIR/dataset_summary.txt" << EOF
이미지 매칭 챌린지 2025 데이터셋 요약
=====================================

총 이미지 수: $total_images

학습 데이터:
- 데이터셋 수: $train_count
- 이미지 수: $train_images

테스트 데이터:
- 데이터셋 수: $test_count
- 이미지 수: $test_images

데이터셋 구조:
$(find "$DATA_DIR" -type d | sort | sed 's/^/- /')

생성 날짜: $(date)
EOF

print_success "데이터셋 요약이 생성되었습니다: $DATA_DIR/dataset_summary.txt"

# 9. 정리
print_step "정리 중..."
echo "다운로드 파일을 삭제하시겠습니까? (y/n)"
read -r cleanup_choice

if [ "$cleanup_choice" = "y" ] || [ "$cleanup_choice" = "Y" ]; then
    echo "다운로드 파일 삭제 중..."
    rm -rf "$DOWNLOAD_DIR"
    print_success "다운로드 파일이 삭제되었습니다."
else
    print_warning "다운로드 파일이 $DOWNLOAD_DIR에 유지됩니다."
fi

# 10. 완료
print_success "데이터셋 설정이 완료되었습니다!"
echo "데이터셋 폴더: $DATA_DIR"
echo "데이터셋 요약: $DATA_DIR/dataset_summary.txt"
echo ""
echo "이제 다음 명령어로 모델 학습을 시작할 수 있습니다:"
echo "./train_image_matching.sh --data_dir $DATA_DIR"
echo ""
echo "감사합니다!"