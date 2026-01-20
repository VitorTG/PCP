import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# EXERCÍCIO 1 - DEMANDA DE SUCO
# ==========================================================

# dados_suco = pd.DataFrame({
#     "month": ["Jan","Fev","Mar","Abr","Mai","Jun","Jul","Ago","Set","Out","Nov","Dez"],
#     "total_demand": [416.00,420.25,405.75,435.75,460.00,487.25,
#                 501.50,473.50,536.25,504.75,525.00,557.00]
# })

# # --- Média móvel simples (6 períodos)
# dados_suco["MM6"] = dados_suco["total_demand"].rolling(6).mean()

# # --- Média móvel ponderada (6 períodos)
# pesos = np.array([0.03, 0.07, 0.15, 0.20, 0.25, 0.30])
# dados_suco["MMP6"] = dados_suco["total_demand"].rolling(6)\
#                      .apply(lambda x: np.dot(x, pesos), raw=True)

# # --- Suavização exponencial
# alfas = [0.2, 0.4, 0.6, 0.8]
# for a in alfas:
#     dados_suco[f"SES_{a}"] = dados_suco["total_demand"].ewm(
#         alpha=a, adjust=False).mean()

# # --- Gráfico
# inicio = dados_suco["MM6"].first_valid_index() + 1
# df_plot = dados_suco.loc[inicio:].reset_index(drop=True)

# plt.figure()

# plt.plot(df_plot["total_demand"], label="Demanda Real")
# plt.plot(df_plot["MM6"], label="MM(6)")
# plt.plot(df_plot["MMP6"], label="MMP(6)")

# for a in alfas:
#     plt.plot(df_plot[f"SES_{a}"], label=f"SES α={a}")

# plt.xticks(
#     ticks=range(len(df_plot)),
#     labels=df_plot["month"]
# )

# plt.title("Exercício 1 - Comparação de Métodos (Julho a Dezembro)")
# plt.xlabel("Meses")
# plt.ylabel("Litros")
# plt.legend()
# plt.grid(True)
# plt.savefig('demand_provision_graph_1')


# # print("\nExercício 1")
# # print(f"Média móvel simples (6): {dados_suco['MM6'].iloc[-1]:.2f} L")
# # print(f"Média móvel ponderada (6): {dados_suco['MMP6'].iloc[-1]:.2f} L")
# # for a in alfas:
# #     print(f"Suavização exponencial (α={a}): {dados_suco[f'SES_{a}'].iloc[-1]:.2f} L")

# print("\n==============================================")
# print("EXERCÍCIO 1 - PREVISÕES MENSAIS (COM DADOS REAIS)")
# print("==============================================\n")

# print("Mês |  Real | MM(6) | MMP(6) | SES α=0.2 | SES α=0.4 | SES α=0.6 | SES α=0.8")
# print("-"*78)

# for i in range(inicio, len(dados_suco)):
#     # Só imprime quando a média móvel existe (Julho em diante)
#     if not np.isnan(dados_suco["MM6"].iloc[i]):

#         print(f"{dados_suco['month'].iloc[i]:>3} | "
#               f"{dados_suco['total_demand'].iloc[i]:5.2f} | "
#               f"{dados_suco['MM6'].iloc[i]:6.2f} | "
#               f"{dados_suco['MMP6'].iloc[i]:7.2f} | "
#               f"{dados_suco['SES_0.2'].iloc[i]:8.2f} | "
#               f"{dados_suco['SES_0.4'].iloc[i]:8.2f} | "
#               f"{dados_suco['SES_0.6'].iloc[i]:8.2f} | "
#               f"{dados_suco['SES_0.8'].iloc[i]:8.2f}")
        

# =============================
# EXERCÍCIO 2 - REGRESSÃO LINEAR
# =============================

# X = np.arange(1, 17)
# Y = np.array([145,147,134,123,177,156,182,180,
#               166,167,188,190,184,181,178,165])

# # Regressão linear
# coef = np.polyfit(X, Y, 1)
# Y_ajust = coef[0]*X + coef[1]

# # Períodos futuros
# X_fut = np.arange(17, 27)
# Y_fut = coef[0]*X_fut + coef[1]

# # --- Gráfico
# plt.figure(figsize=(10, 6))

# # Pontos reais + linha tracejada
# plt.plot(X, Y, 'o--', label="Demanda Real")

# # Linha da regressão
# plt.plot(X, Y_ajust, '-', label="Regressão Linear")

# # Projeção futura
# plt.plot(X_fut, Y_fut, '--', label="Previsão Futura")

# plt.title("Exercício 2 - Regressão Linear")
# plt.xlabel("Período")
# plt.ylabel("Demanda")
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.savefig("demand_provision_graph_2")
# plt.close()
# print("\n==============================================")
# print("EXERCÍCIO 2 - PREVISÃO DE DEMANDA (10 PERÍODOS)")
# print("==============================================\n")

# print("Período | Demanda Prevista")
# print("-" * 28)

# for periodo, demanda in zip(X_fut, Y_fut):
#     print(f"{periodo:7d} | {demanda:16.2f}")

# ==========================================================
# EXERCÍCIO 3 - HOTEL (MÉDIA MÓVEL 3 SEMANAS)
# ==========================================================

# hospedes = [400, 380, 411]
# prev_sem4 = np.mean(hospedes)
# prev_sem5 = np.mean([380, 411, 415])
# erro_sem4 = 415 - prev_sem4

# print("\n==============================================")
# print("EXERCÍCIO 3 - PREVISÃO DE DEMANDA HOTEL")
# print("==============================================\n")
# print(f"Previsão semana 4: {prev_sem4:.2f}")
# print(f"Previsão semana 5: {prev_sem5:.2f}")
# print(f"Erro semana 4: {erro_sem4:.2f}")

# ==========================================================
# EXERCÍCIO 4 - IMPRESSORAS (MÉDIA PONDERADA)
# ==========================================================

# imp = [100, 90, 105, 95]

# prev_mes5 = 0.4*95 + 0.3*105 + 0.2*90 + 0.1*100
# prev_mes6 = 0.4*110 + 0.3*95 + 0.2*105 + 0.1*90

# print("\n==============================================")
# print("EXERCÍCIO 4 - PREVISÃO DE DEMANDA IMPRESSORAS")
# print("==============================================\n")
# print(f"Previsão mês 5: {prev_mes5:.2f}")
# print(f"Previsão mês 6: {prev_mes6:.2f}")

# # =============================
# # EXERCÍCIO 5 - SUAVIZAÇÃO EXPONENCIAL
# # =============================

# # Demandas reais (índice = mês)
# demanda_real = {
#     1: 450,
#     2: 505,
#     3: 516,
#     4: 488,
#     5: 467,
#     6: 554,
#     7: 510
# }

# previsao_inicial = 500
# alfas = [0.1, 0.8]

# # Dicionário para armazenar previsões
# previsoes = {}

# # -----------------------------
# # Cálculo das previsões
# # -----------------------------
# for a in alfas:
#     F = {}
#     F[1] = previsao_inicial

#     for mes in range(2, 9):  # até mês 8
#         if mes - 1 in demanda_real:
#             F[mes] = a * demanda_real[mes - 1] + (1 - a) * F[mes - 1]
#         else:
#             F[mes] = F[mes - 1]  # quando não há demanda real

#     previsoes[a] = F

# print("\n==============================================")
# print("EXERCÍCIO 5 (a) - PREVISÃO PARA O MÊS 2")
# print("==============================================\n")

# print("Alpha | Previsão Mês 2")
# print("-" * 26)

# for a in alfas:
#     print(f"{a:5.1f} | {previsoes[a][2]:15.2f}")

# print("\n==============================================")
# print("EXERCÍCIO 5 (b) - PREVISÕES DE DEMANDA ATÉ O MÊS 8")
# print("==============================================\n")

# print("Mês | Alpha 0.1 | Alpha 0.8")
# print("-" * 30)

# for mes in range(2, 9):
#     print(f"{mes:3d} | {previsoes[0.1][mes]:9.2f} | {previsoes[0.8][mes]:9.2f}")

# import matplotlib.pyplot as plt

# # Meses onde existe demanda real
# meses = list(range(1, 8))
# demanda = [demanda_real[m] for m in meses]

# prev_01 = [previsoes[0.1][m] for m in meses]
# prev_08 = [previsoes[0.8][m] for m in meses]

# plt.figure(figsize=(9, 5))

# plt.plot(meses, demanda, 'o-', label="Demanda Real")
# plt.plot(meses, prev_01, 's--', label="Previsão α = 0.1")
# plt.plot(meses, prev_08, 'd--', label="Previsão α = 0.8")

# plt.title("Exercício 5 - Análise das Previsões por Suavização Exponencial")
# plt.xlabel("Mês")
# plt.ylabel("Demanda")
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.savefig("demand_provision_graph_3", dpi=300)
# plt.close()


# # =============================
# # EXERCÍCIO 6 - SUAVIZAÇÃO EXPONENCIAL
# # =============================

# alpha = 0.4

# # Dados conhecidos
# F_jan = 950
# D_jan = 820
# D_fev = 980

# # -----------------------------
# # a) Previsão para Fevereiro
# # -----------------------------
# F_fev = alpha * D_jan + (1 - alpha) * F_jan

# print("\n==============================================")
# print("EXERCÍCIO 6 (a) - PREVISÃO PARA FEVEREIRO")
# print("==============================================\n")

# print("Mês        | Previsão")
# print("-" * 28)
# print(f"Fevereiro | {F_fev:8.2f}")

# # -----------------------------
# # b) Previsão para Março
# # -----------------------------
# F_mar = alpha * D_fev + (1 - alpha) * F_fev

# # Erro de Fevereiro
# erro_fev = abs(D_fev - F_fev)

# print("\n==============================================")
# print("EXERCÍCIO 6 (b) - PREVISÃO PARA MARÇO E ERRO")
# print("==============================================\n")

# print("Mês       | Valor")
# print("-" * 28)
# print(f"Fevereiro | Previsão = {F_fev:8.2f}")
# print(f"Fevereiro | Real     = {D_fev:8.2f}")
# print(f"Fevereiro | Erro     = {erro_fev:8.2f}")
# print("-" * 28)
# print(f"Março     | Previsão = {F_mar:8.2f}")


# # =============================
# # EXERCÍCIO 7 (a) - CORRELAÇÃO
# # =============================

# demanda_cimento = np.array([735, 600, 770, 670, 690, 780, 640])
# taxa_construcao = np.array([100, 80, 105, 92, 95, 107, 87])

# # Cálculo do coeficiente de correlação de Pearson
# correlacao = np.corrcoef(taxa_construcao, demanda_cimento)[0, 1]

# print("\n==============================================")
# print("EXERCÍCIO 7 (a) - COEFICIENTE DE CORRELAÇÃO")
# print("==============================================\n")

# print("Coeficiente de correlação (r)")
# print("-" * 34)
# print(f"r = {correlacao:.3f}")

# # =============================
# # EXERCÍCIO 8 - AJUSTE DE TENDÊNCIA
# # =============================

# anos_hist = np.array([88, 89, 90, 91, 92, 93, 94, 95, 96, 97])
# demanda_hist = np.array([250, 230, 270, 285, 290, 287, 310, 325, 320, 340])

# # Cálculo da linha de tendência (regressão linear)
# coef = np.polyfit(anos_hist, demanda_hist, 1)
# tendencia_linear = coef[0] * anos_hist + coef[1]

# # Gráfico
# plt.figure(figsize=(9, 5))

# plt.plot(anos_hist, demanda_hist, 'o-', label="Demanda Histórica")
# plt.plot(anos_hist, tendencia_linear, '--', label="Linha de Tendência")

# plt.title("Exercício 8 (a) - Demanda Histórica e Tendência")
# plt.xlabel("Ano")
# plt.ylabel("Demanda")
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.savefig("demand_provision_graph_4")
# plt.close()

# # -----------------------------
# # Estimativa inicial da tendência
# # -----------------------------
# tendencia_inicial = np.mean([
#     demanda_hist[1] - demanda_hist[0],
#     demanda_hist[2] - demanda_hist[1],
#     demanda_hist[3] - demanda_hist[2]
# ])

# nivel_inicial = demanda_hist[3]  # demanda_hist do 4º ano

# alpha1 = 0.1
# alpha2 = 0.2

# nivel = nivel_inicial
# tendencia = tendencia_inicial

# # Ajuste do modelo até o último ano conhecido (96)
# for i in range(4, len(demanda_hist)):
#     novo_nivel = alpha1 * demanda_hist[i] + (1 - alpha1) * (nivel + tendencia)
#     nova_tendencia = alpha2 * (novo_nivel - nivel) + (1 - alpha2) * tendencia

#     nivel = novo_nivel
#     tendencia = nova_tendencia

# # Previsão para 1997
# previsao_97 = nivel + tendencia

# print("\n==============================================")
# print("EXERCÍCIO 8 (b) - PREVISÃO PARA O ANO DE 1997")
# print("==============================================\n")

# print(f"Tendência inicial estimada : {tendencia_inicial:.2f}")
# print(f"Nível final estimado       : {nivel:.2f}")
# print(f"Tendência final estimada   : {tendencia:.2f}")
# print("-" * 38)
# print(f"Previsão da demanda em 97  : {previsao_97:.2f}")


# =============================
# EXERCÍCIO 9 (a) - SAZONALIDADE E TENDÊNCIA
# =============================

# Demandas observadas
demanda = np.array([65, 58, 50, 60, 85, 75, 62, 74])

# Trimestres correspondentes
trimestres = np.array([1, 2, 3, 4, 1, 2, 3, 4])

# Índices de sazonalidade
indice_sazonal = {
    1: 1.3,
    2: 1.0,
    3: 0.8,
    4: 0.9
}

# -----------------------------
# Retirada da sazonalidade
# -----------------------------
demanda_dessazonalizada = np.array([
    demanda[i] / indice_sazonal[trimestres[i]]
    for i in range(len(demanda))
])

# -----------------------------
# Ajuste da tendência (regressão linear)
# -----------------------------
periodos = np.arange(1, len(demanda) + 1)

coef = np.polyfit(periodos, demanda_dessazonalizada, 1)
a = coef[0]  # inclinação
b = coef[1]  # intercepto


print("\n==============================================")
print("EXERCÍCIO 9 (a) - RETIRADA DA SAZONALIDADE")
print("==============================================\n")

print("Período | Trim | Demanda | Índice | Dessazonalizada")
print("-" * 54)

for i in range(len(demanda)):
    print(f"{i+1:7d} | {trimestres[i]:4d} | "
          f"{demanda[i]:8.2f} | "
          f"{indice_sazonal[trimestres[i]]:6.2f} | "
          f"{demanda_dessazonalizada[i]:14.2f}")
