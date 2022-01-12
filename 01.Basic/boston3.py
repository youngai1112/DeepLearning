# 입력으로 0에서 100 사이의 정수값을 받아서
# 보스톤 주택가격의 실제값과 예측값을 보여주는 프로그램
# 모델을 학습하지 않고 저장된 모델 활용
# 사용법: python boston3.py 20 (test_dataset_index)



# import part
import sys
import numpy as np
# import warnings
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model


# argument 정리 
if len(sys.argv) <= 1:          # argument 미입력
    print(sys.argv[0], len(sys.argv))
    print('사용법: python boston3.py test_dataset_index(0~50) 2> /dev/null')
    sys.exit()          
try: 
    index = int(sys.argv[1])    # 여기서 에러가 난다면, 즉 정수를 입력한 것이 아니라면 ( 정수입력시 통과 )        
except: 
    print('정수를 입력하세요')  # 문자열 입력시
    print('사용법: python boston3.py test_dataset_index(0~50) 2> /dev/null')
    sys.exit()
if index < 0 or index > 50:     # 입력받고자 하는 숫자를 넘겨서 입력했을 때
    print('0과 50사이의 정수를 입력하세요')
    print('사용법: python boston3.py test_dataset_index(0~50) 2> /dev/null')
    sys.exit()


# 상수값 설정 등 변수 초기화
seed = 2022
model_filename = 'boston.h5'
# warnings.filterwarnings('ignore')
np.random.seed(seed)
boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(
    boston.data, boston.target, test_size=0.1, random_state=seed
)


# 메인 모델 만들기: 반복되는 과정이므로 학습한 모델을 save해서 모델을 가져가서 구현할 수 있다. 
# 학습한 모델을 파일화 또는 디스크에 저장해서 필요할 때 사용하는 것이 훨씬 효율적이다
model = load_model(model_filename)


# 입력값 받기
# index = int(input('0 ~ 50 정수값을 입력하세요.> '))
test = X_test[index].reshape(1,-1)
pred_value = model.predict(test)


# 최종결과 출력
print(f'실제값:{y_test[index]}, 예측값:{pred_value[0,0]:.2f}')