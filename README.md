# laxus
Location-Aware Update of Edge Infrastructures.


Laxus segue a mesma ideia do Salus, migrando somente aplicações de servidores não atualizados
    - Em um cromossomo C, o gene C_n representa a n-ésima aplicação hospedada por um servidor não atualizado


Objetivos:
    - Minimizar número de servidores não atualizados hospedando aplicações
    - Minimizar número de migrações que não geram esvaziamento de servidores não atualizados ou que colocam aplicações em servidores não atualizados
    - Minimizar número de violações de SLA


## LISTA DE TAREFAS
Criar gráficos



## Instalação

sudo apt install libblas3 liblapack3 liblapack-dev libblas-dev

## Execução

reset && python3 -B -m simulator --dataset dataset2 --algorithm salus && python3 -B -m simulator --dataset dataset2 --algorithm greedy_least_batch && python3 -B -m simulator --dataset dataset2 --algorithm first_fit_like && python3 -B -m simulator --dataset dataset2 --algorithm worst_fit_like && python3 -B -m simulator --dataset dataset2 --algorithm best_fit_like