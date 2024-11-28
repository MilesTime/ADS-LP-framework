import pulp
import numpy as np
import time
import os

num_cores = os.cpu_count()
print(f"Number of CPU cores: {num_cores}")

# Initialization settings for values
T = 5  # Total time periods
Q = 400  # Number of order types
M = 700  # Number of methods
I = 30  # Number of item types
K = 10  # Number of facilities
np.random.seed(0)
C = np.random.rand(M * Q).astype(np.float32)  # Cost vector, using float32
A = np.zeros((Q, M * Q), dtype=np.float32)  # A matrix, using float32
E = np.zeros((I * K, M * Q), dtype=np.float32)  # E matrix, using float32
b = np.random.rand(Q).astype(np.float32)  # Demand vector, using float32
d = np.random.randint(1, Q/2, (I * K, 1)).astype(np.float32)  # Inventory capacity vector, using float32
k = Q
L_bits = (M * Q + Q + Q * M * Q + I * K * M * Q + I * K) * 32
print(f"Total bit-length L for the problem encoding is: {L_bits} bits")

# Inventory for discard methods corresponds to the warehouse, warehouse 0, the first I are (I,0) corresponding to the discard warehouse inventory
for i in range(I):
    d[i] = 1e6  # Modify only the first facility of each item

# Costs corresponding to discard methods, discard methods are randomly chosen, each q corresponds to a high-cost method
discard_method_indices = np.random.choice(range(M), Q, replace=False)
for q in range(Q):
    idx = discard_method_indices[q]
    C[idx + q * M] = k * C[idx + q * M]
# Generate random items for each order type
# Generate random I for each q
items_per_q = {q: np.random.choice(range(I), np.random.randint(1, I+1), replace=False) for q in range(Q)}

m_ik = {}
m_i = {}

# Set up discard methods, each q corresponds to one, the warehouse corresponding to the discard method is k=0. The elements contained in the discard method are exactly the i corresponding to q. At the same time, set the corresponding positions of A and E to 1.
for q, idx in enumerate(discard_method_indices):
    # print(q, idx)  # q: order, idx: discard method corresponding to order q
    i_k_pairs = [(i, 0) for i in items_per_q[q]]  # i_k_pairs: discard method m corresponding to (i, 0)
    m_ik[idx] = i_k_pairs
    m_i[idx] = [pair[0] for pair in i_k_pairs]
    A[q, idx + q * M] = 1
    for i, k in i_k_pairs:
        E[i + k * I, idx + q * M] = 1

# Generate corresponding (i, k) for each method and save the i used in each method
for m in range(M):
    if m not in discard_method_indices:
        num_i = np.random.randint(1, I + 1)  # Generate a random number of items in the range [1, I]
        i_k_pairs = [(np.random.randint(0, I), np.random.randint(1, K)) for _ in range(num_i)]  # Random pairs of items and warehouses
        m_ik[m] = i_k_pairs
        m_i[m] = [pair[0] for pair in i_k_pairs]

# Set up A matrix. Rule: if all items in method m are a subset of items in q (representing m ~ q), then set the corresponding element in A to 1.
# Set up E matrix. Rule: for each order q, set (i, k) pairs contained in methods m that can be used to 1.
for q in range(Q):
    for m in range(M):
        m_index = m + q * M
        if all(item in items_per_q[q] for item in m_i[m]):
            A[q, m_index] = 1
        for i, k in m_ik[m]:
            E[i + k * I, m + q * M] = 1

path_to_cplex = "/Applications/CPLEX_Studio2211/cplex/bin/x86-64_osx/cplex"

def Orginal_resource_allocation(M, Q, I, K, C, b, A, E, d):
    start_time = time.time()
    problem = pulp.LpProblem("Resource_Allocation", pulp.LpMinimize)
    solver = pulp.CPLEX_CMD(path=path_to_cplex,options=['set lpmethod 4'])
    X = pulp.LpVariable.dicts("X", range(M * Q), 0)
    problem += pulp.lpSum([C[mq] * X[mq] for mq in range(M * Q)])
    for q in range(Q):
        problem += pulp.lpSum([X[m + q * M] * A[q, m + q * M] for m in range(M)]) == b[q]
    for ik in range(I * K):
        problem += pulp.lpSum([X[m + q * M] * E[ik, m + q * M] for q in range(Q) for m in range(M)]) <= d[ik]
    model_build_time = time.time() - start_time
    status = problem.solve(solver)
    solve_time = time.time() - model_build_time - start_time
    OPT = pulp.value(problem.objective)
    result_processing_time = time.time() - solve_time - model_build_time - start_time

    total_execution_time = time.time() - start_time
    return {
        "solve_time": solve_time,
        "optimal_value": OPT,
        "status": pulp.LpStatus[status]
    }

result = Orginal_resource_allocation(M, Q, I, K, C, b, A, E, d)
print(result)

def solve_lagrangian_relaxation_batch(I, K, M, Q, C, b, d, E, A, J, alpha_J, batch_size):
    lambda_ = np.zeros(I * K)
    batch_results = []
    batch_solve_times = []
    batch_iteration_times = []

    def solve_Lq_batch(lambda_vec, batch_index):
        start_time = time.time()
        solver = pulp.CPLEX_CMD(path=path_to_cplex, options=['set lpmethod 4'])
        L_q_batch = pulp.LpProblem(f"Lagrangian_batch_{batch_index}", pulp.LpMinimize)
        total_objective = pulp.LpAffineExpression()

        batch_sol = {}
        for q in range(batch_index * batch_size, min((batch_index + 1) * batch_size, Q)):
            indices = [i for i in range(M * Q) if A[q][i] == 1]
            X_q = pulp.LpVariable.dicts(f"X_{q}", indices, 0, 1, cat=pulp.LpContinuous)
            cost_component = pulp.lpSum([C[i] * X_q[i] for i in indices])
            penalty_component = pulp.lpSum([lambda_vec[ik] * E[ik, i] * X_q[i] for ik in range(I * K) for i in indices])
            total_objective += b[q] * (cost_component + penalty_component)
            L_q_batch += pulp.lpSum([X_q[i] for i in indices]) == 1
            batch_sol[q] = X_q

        L_q_batch.setObjective(total_objective)
        model_build_time = time.time() - start_time
        L_q_batch.solve(solver)
        end_time = time.time()

        for q in batch_sol:
            indices = [i for i in range(M * Q) if A[q][i] == 1]
            batch_sol[q] = {i: pulp.value(batch_sol[q][i]) for i in indices}
        batch_solve_times.append((batch_index, end_time - model_build_time - start_time))
        return batch_sol

    overall_start_time = time.time()

    for j in range(J):
        iteration_start_time = time.time()
        X_prime = {batch_index: solve_Lq_batch(lambda_, batch_index) for batch_index in range((Q + batch_size - 1) // batch_size)}
        G = np.zeros(I * K)
        for batch_index in X_prime:
            for q in X_prime[batch_index]:
                for ik in range(I * K):
                    G[ik] += sum(E[ik, i] * X_prime[batch_index][q].get(i, 0) for i in range(M * Q) if A[q][i] == 1)
        G -= d.flatten()
        lambda_ = np.maximum(0, lambda_ + alpha_J * G)
        batch_iteration_times.append((j, time.time() - iteration_start_time))
        batch_results.append(X_prime)

    overall_time = time.time() - overall_start_time

    return {
        "batch_results": batch_results,
        "batch_solve_times": batch_solve_times,
        "batch_iteration_times": batch_iteration_times
    }

def calculate_performance_batch(results, M, Q, C, b, d, T, I, K, OPT, k, J, alpha_J):
    batch_average_solutions = np.zeros(M * Q)
    for j in range(J):
        batch_tmp_solution = np.zeros(M * Q)
        for batch_index in results['batch_results'][j]:
            for q in results['batch_results'][j][batch_index]:
                current_solution = np.array([results['batch_results'][j][batch_index][q].get(i, 0) for i in range(M * Q)])
                batch_average_solutions += b[q] * current_solution
                batch_tmp_solution += b[q] * current_solution

        print(f"Iteration {j} time tmp solution Max Inventory capacity constraint violation:", np.min(d - np.dot(E, batch_tmp_solution)))
        print(f"Iteration {j} time tmp objective function:", np.sum(np.dot(C, batch_tmp_solution)))

    total_Lq_batch_time.append(sum([time for _, time in results["batch_solve_times"]]))
    avg_Lq_batch_time.append(sum([time for _, time in results["batch_solve_times"]]) / (J * Q))
    OPT_batch_time.append(result["solve_time"])
    print("total_Lq_batch_time", sum([time for _, time in results["batch_solve_times"]]))
    print("avg_Lq_batch_time", sum([time for _, time in results["batch_solve_times"]]) / (J * Q))
    theorical_total_batch_time_upbound.append((M * Q) ** 3.5)
    theorical_total_L_q_batch_time_upbound.append(J * Q * ((M) ** 3.5))
    batch_average_solutions /= J
    print("arrival prob constraint violation:", np.dot(A, batch_average_solutions) - b)
    print("Max Inventory capacity constraint violation:", np.min(d - np.dot(E, batch_average_solutions)))
    batch_total.append(batch_average_solutions)
    batch_dot_product = np.dot(C, batch_average_solutions)
    print("batch_average_solutions", batch_dot_product)
    batch_total_cost.append(batch_dot_product)

    q_max = max(len(results["batch_results"][j][batch_index][q]) for j in range(J) for batch_index in results["batch_results"][j] for q in results["batch_results"][j][batch_index])
    C_lowerbound = (q_max * T * T + max(d) ** 2 * I * K) / OPT
    C_bar_lowerbound = ((2 + C_lowerbound * alpha_J) * OPT) ** (1 / 2)

    batch_performance_upperbound_2.append((1 + C_lowerbound / (2 * J ** (1 / 3))) * OPT)
    batch_performance_upperbound_3.append((1 + (k - 1) * C_bar_lowerbound / (min(d) * alpha_J * (J ** (1 / 2)) + C_bar_lowerbound)) * (1 + C_lowerbound / (2 * J ** (1 / 3))) * OPT)

# Run algorithm and performance evaluation
batch_total = []
batch_performance_upperbound_2 = []
batch_performance_upperbound_3 = []
batch_total_cost = []
total_Lq_batch_time = []
avg_Lq_batch_time = []
OPT_batch_time = []
Approx_y_batch = []
theorical_total_batch_time_upbound = []
theorical_total_L_q_batch_time_upbound = []
OPT = result["optimal_value"]
J_values = [5, 10, 20, 50]
OPT_value = [OPT for _ in range(len(J_values))]

for J in J_values:
    alpha_J = J ** (-1 / 3)
    batch_size = 5  # Set batch size to 1 q
    L_q_result_batch = solve_lagrangian_relaxation_batch(I, K, M, Q, C, b, d, E, A, J, alpha_J, batch_size)
    calculate_performance_batch(L_q_result_batch, M, Q, C, b, d, T, I, K, OPT, k, J, alpha_J)

# Approximation part
discard_method = {q: discard_method_indices[q] for q in range(Q)}

for i in batch_total:
    inventory_violations = E.dot(i) / d.flatten()  # Calculate the degree of violation for each inventory
    tau = max(1, np.max(inventory_violations))
    print("Approximation degree", tau)
    y = np.zeros(M * Q)
    for q in range(Q):
        for n in range(q * M, (q + 1) * M):
            if n % M != discard_method[q]:
                y[n] = i[n] * (1 / tau)
        for n in range(q * M, (q + 1) * M):
            if n % M == discard_method[q]:
                total_x = np.sum(y[q * M:(q + 1) * M])
                y[n] = b[q] - total_x + i[n]
    Approx_y_batch.append(np.dot(C, y))