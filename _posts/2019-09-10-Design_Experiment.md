---
layout: post
title: Design and Analysis of Experiment(1) Randomization
---

이 포스트는 동국대학교 박주현 교수님의 '실험계획법' 과목을 수강하면서 혼자 복습하고 정리하기 위해서 올리는 포스트입니다. 포스트에서 사용되는 R-code는 교수님의 코드를 참고하였음을 밝힙니다.

### radomization

* 정의: randomization은 크게 2가지로 나뉠 수 있습니다. 첫째, units에게 treatment를 할당하는 방법에 쓰일 수 있으며, 둘째, response를 평가받는 units의 순서를 정할 때 쓰일 수 있습니다. 우리는 주로 첫번째 경우를 많이 보게 될 것입니다. 실험이 randomized되었다.' 라고 말할 수 있는 경우는 treatments를 units에게 할당하는 method가 known and well-understood probablistic scheme를 포함하고 있을 때입니다. 

 randomization과 haphazard는 구분되어야 합니다. 예를 들어 설명하겠습니다. 여기 unit에게 treatments를 할당하는 2가지 방법이 있습니다. ==**방법1**==은 16명의 사람들에게 4가지 treatment를 할당하려고 합니다. 이 때, 같은 재질의 종이에 4가지 treatment의 종류를 각각 4번씩 적은 종이 16장을 바구니에 넣고 마구마구 섞습니다. 그 다음 16명의 사람은 바구니에서 각각 한 장의 종이를 뽑고, 그 종이에 적힌 treatment를 할당받습니다. ==**방법2**==는 treatment_A는 investigator가 마주치는 첫 4명의 사람들에게 부여하고, treatment_B는 그다음 마주치는 4명의 사람들에게.. 이런식으로 4개의 treatment를 사람들에게 부여합니다.
 ==**방법1**==의 경우, 우리는 radnomized되었다고 말할 수 있고 ==**방법2**==는 haphazard이므로 그렇지 않다라고 할 수 있습니다. 왜냐하면 ==**방법1**==은 우리가 다른 실험에서도 그와 같은 방식으로 treatment를 할당할 수 있습니다. 또, 같은 실험에서도 여러개의 표본을 추출할 수 있습니다. 그러나 ==**방법2**== 는 method가 아니여서 그와 같은 방법으로 다른 실험에 적용할 수 없습니다. 이런 경우를 '체계적이지 않고 이해가능하지 않다'고 책에서 표현합니다. 
 

* 목적: radomization을 하는 이유는 크게 2가지 입니다. 첫째, confounding을 효과적으로 방지해줍니다. confounding factor란 treatment가 아닌 다른 factor가 response에 영향을 주어 response를 해석할 때 treatment효과와 다른 factor 간 분리를 할 수 없는 경우, 그 다른 factor를 confounding factor라 합니다. 둘째, randomization은 여러 inference의 basis가 됩니다. 

* 예시: 다음의 paired t-test 실험을 생각해봅시다. 우리는 이 실험을 통해 사무실 환경이 standard할 때와 ergonomic할 때 workers(units)들의 업무처리속도를 비교하고자 합니다. 여기서 alternative hypothesis은 'standard평균-erogonomic평균>0'입니다. 여기서 unit은 총 30명이라 합시다. 각 유닛에서 *standard평균-erogonomic평균*의 절댓값을 취해주고, *standard평균-erogonomic평균*값에 +,-의 부호 2개가 할당될 확률이 likely equaly라고 가정합니다. 그렇다면, 이 경우 우리가 고려해야할 모든 경우의 수는 2^30가지 입니다... 뭐 일단 다 계산했다고 칩시다. 그렇다면 이제 descriptive statistic for the data와 p-value를 구해야겠죠.  descriptive statistic은 이 실험의 경우 sum of the differences입니다. pvalue를 구하려면 under null에서의 descriptive statistic(각 경우에서의 차이의 합)의 분포를 알아야 합니다. 이 경우 우리는 총 2의 30승의 경우가 있으므로 2의 30승 개의 통계량이 있고, 이를 histogram으로 그러면 거의 정규분포에 근사하게 됨을 알 수 있습니다. 이는 central limit theorem때문이라고 볼 수 있겠죠. 이 분포의 통계량이 '평균'으로 나타날 수 있기 때문입니다. (CLT는 확률변수의 분포에 상관없이, 그 확률변수의 표본평균은 sample이 무한대로 커지면 정규분포로 근사한다는 것을 의미합니다.) observed value(test statistic)은 2의 30승의 모든 경우 중에서 실제 우리가 unit로 부터 관측된 sum of the differences값이고, 자료에서는 0.23이라 합니다. 그러면 p-value는 0.23과 같거나 큰 경우(우단측검정이므로.)의 수를 2의 30승에서 찾아서 분자/분모로 취해준 확률값이 되겠습니다. p-value는 이 경우 0.069입니다. 그런데 unit=10인 경우 p-value는 0.454이고, unit=30인경우는 p-value가 0.072로 좀 애매하다고 할 수 있죠. 왜그럴까요? 우선 밑의 테이블을 봅시다. 
<table>
  <thead>
    <tr>
      <th></th>
      <th>t_test</th>
      <th>randomization test</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>p-value(unit=10)</td>
      <td>0.459</td>
      <td>0.454</td>
    </tr>
    <tr>
      <td>p-value(unit=30)</td>
      <td>0.074</td>
      <td>0.072</td>
    </tr>
  </tbody>
</table>

 결론부터 말하면 t-test의 결과와 randomization test결과는 거의 유사하므로 실제 실험에서는 t-test를 거의 사용합니다. 그런데 unit이 크면 왜 p-value가 작아져서 reject null을 할 가능성이 조금 더 커질까요? 이는 sample size가 커지면 분산이 작아져서 p-value가 작아지는 원리로 설명할 수 있겠네요.

* 한계점: 
 1. 실험은 크게 2가지로 분리할 수 있는데, experience experiment와 observation experiment입니다. 보통 전자는 unit이 사람이 아닌 경우 후자는 사람인 경우를 주로 칭합니다. unit이 사람인 경우, treatment를 할당하는 것이 윤리적으로 위배될 경우가 많이 존재하기 때문에, 보통 observation experience를 하게 됩니다.  
 
 2. 어떤 실험에 대해 randomize를 할 때, 그 경우의 수가 매우 많아 계산량이 큽니다. 예를 들면, 8명을 2개 집단으로 나누어 각 집단의 4명씩에게 어떤 값을 할당한다고 합시다. 이렇게 간단해 보이는 실험에서도 총 40320개의 경우의 수가 발생합니다. 이 모든 경우를 계산하여  p-value와 statistic을 비교하는 것은 시간이 많이 걸릴 것입니다. 

* 극복방안: 실제로 실험에서는 randomization 대신 t-test나 anova를 사용하는 경우가 대다수입니다. 물론, t-test나 anova 모두 표본그룹의 독립성, 정규성, 등분산성이라는 가정을 만족하는 데이터를 가지고 있어야겠죠. 그러나 만약 가정을 따르지 않는다면? The most powerful test( MP TEST)를 하게 됩니다. 이는 한가지 실험가설에 대하여 여러가지 검정방법을 실시합니다. 그리고 각 검정방법마다 alpha값을 고정시킨 다음, power가 큰 검정방법을 택하는 것입니다. 예를 들면, 3개의 그룹의 평균을 비교할 때, 우리는 anova와 k-wallis 2가지를 해본다고 합시다. 이 경우 데이터가 가정을 만족하지 않아도 anova가 sample size가 dramatic하게 작지 않는 이상 k-wallis방법보다 power가 항상 큽니다. 따라서 우리는 데이터가 가정을 만족하든 안하든 anova를 먼저 실시해보는 것입니다. randomization도 마찬가지겠죠. 앞서 봤듯이 t-test의 pvalue와 거의 유사한 결과를 도출했으니까요.

* 참고: power란 무엇일까요? 일단 1종오류는 집단 간 차이가 없는데 있다고 우리가 판단한 경우(대립가설이 틀린데 맞다고 판단한 경우)의 확률입니다. 2종오류는 반대로 실제 차이가 있는데 없다고 판단할 확률이죠. 따라고 1-2종오류=power란 실제 차이가 있다면, 실제 차이가 있다고 우리가 판단할 확률을 말합니다. power가 80%라 함은 100번 실험을 했을 때 이 중 80번은 그 차이를 우리가 찾아낼 수 있다는 말과 동치를 이룹니다. power은 sample size와 같이 해석해야합니다. sample size가 크면, 앞서 표를 보셨듯이 pvalue가 작아져 미묘한 차이도 차이라고 말할 수 있습니다. 이 때, 설계자들은 effect size와 statistical significance 2가지를 모두 고려하여 sample size를 측정해야 겠습니다. 혈압약 임상실험에서 혈압이 1 떨어졌다면 그 혈압약을 statistical sign.쪽은 차이가 있다고 하겠지만 실제 effect side에서는 콧웃음을 치겠죠. 보통 20이상은 차이나야 유의미하다고 보겠죠. 이 2가지 관점을 잘 생각해서 size를 설정해야하는 것도 실험계획의 일부입니다.

### r code simulation
```bash
x <- c(4.3,4.6,4.8,5.4)
y <- c(5.3,5.7,6.0,6.3)
# t. test() function
#### parameter: x,y => numeric vector of data values
#### alternative: character string(options= two.sided, greater,less)
#### 즉 대립가설의 방향이 양측인지(default), 우단측인지 좌단측인지 결정,
#### greater => x의 평균>y의평균 (우단측검정) 
#### paired= if want paired t.test , var.equal: T,F=> decide whether to treat 
#### two variances as being equal. True로 설정하면 pooled variance, 아니라면 
#### welch approximation to the dgree of freedom이 사용된다. default=False 

t.test(x,y) # 양측검정 
t.test(x,y,alternative="greater",var.equal=TRUE) # 우단측검정, 등분산가정
# pt function
### pt(q,df,ncp,lower.tail=TRUE,..)
### pt(t-statistic, 자유도,lower.tail=TRUE라면 p(X<=q)의확률값을 계산한다. 

pt(-3.3273,6,lower.tail=FALSE) # same p-value above. (등분산가정,우단측 t test결과와 동일)

ns<-length(c(x,y)) # x벡터 원소갯수+y벡터 원소갯수=8

set.seed(101)
n.itr<-10000 # 반복할 실험횟수= 만번
diff.vec<-rep(NA, n.itr) ## NA가 만개있는 벡터. 여기에 한번실험해서 나온 평균차이를 입력.

for (j in 1:n.itr){
 # sample.int(ns까지 자연수에서 표본추출,표본크기)
x.indx<-sample.int(ns, length(x))
y.indx<-(1:ns)[-x.indx]
diff.vec[j]<-mean(c(x,y)[x.indx])-mean(c(x,y)[y.indx])
}
sum(diff.vec>(mean(x)-mean(y)))/n.itr
```
