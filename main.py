import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random

# Parámetros
num_usuarios = 5
num_routers = 2
num_servidores = 8
num_iteraciones = 50
alpha = 0.1  # Tasa de aprendizaje
gamma = 0.9  # Factor de descuento
epsilon = 0.2  # Exploración vs explotación
server_processing_time = 5  # Tiempo de procesamiento por solicitud en cada servidor

# Crear grafo
G = nx.DiGraph()
usuarios = [f"U{i}" for i in range(num_usuarios)]
routers = [f"R{i}" for i in range(num_routers)]
servidores = [f"S{i}" for i in range(num_servidores)]

# Agregar nodos
G.add_nodes_from(usuarios, type='usuario')
G.add_nodes_from(routers, type='router')
G.add_nodes_from(servidores, type='servidor')

# Conexiones
edges = []
for u in usuarios:
    r = random.choice(routers)  # Cada usuario se conecta a un router aleatorio
    edges.append((u, r))
for r in routers:
    for s in servidores:
        edges.append((r, s))
G.add_edges_from(edges)

# Posiciones
pos = {}
pos.update({u: (0, i) for i, u in enumerate(usuarios)})
pos.update({r: (1, i + 0.5) for i, r in enumerate(routers)})
pos.update({s: (2, i) for i, s in enumerate(servidores)})

# Inicialización de tráfico
traffic = {edge: 0 for edge in edges}
server_load = {s: 0 for s in servidores}
router_traffic = {r: 0 for r in routers}  # Contador de tráfico por router
server_timers = {s: 0 for s in servidores}  # Temporizadores para cada servidor

# Inicialización de tablas Q
Q = {r: {s: 0 for s in servidores} for r in routers}

# Variables para análisis
traffic_history = []
q_values_history = {r: [] for r in routers}
server_load_history = {s: [] for s in servidores}
router_traffic_history = {r: [] for r in routers}
reward_history = []

def choose_server(router):
    if random.random() < epsilon:
        return random.choice(servidores)  # Exploración
    return max(Q[router], key=Q[router].get)  # Explotación

# Función para balancear el tráfico entre routers
def balance_router_traffic():
    min_traffic_router = min(router_traffic, key=router_traffic.get)  # Router con menos tráfico
    return min_traffic_router

# Función de actualización para la animación con recompensa ajustada y balanceo de tráfico
def update(frame):
    plt.clf()
    
    # Generar tráfico y actualizar Q-learning
    total_reward = 0
    for u in usuarios:
        if random.random() < 0.3:  # Probabilidad de enviar solicitud
            router = balance_router_traffic()  # Elegir el router con menos tráfico
            server = choose_balanced_server(router)  # Función mejorada para elegir el servidor
            
            # Actualizar carga del servidor
            server_load[server] += 1
            router_traffic[router] += 1  # Incrementar tráfico del router
            traffic[(router, server)] += 1
            
            # Recompensa ajustada
            # Penalizar la carga alta
            if server_load[server] > 5:  # Umbral de carga alta
                reward = -10  # Penalización fuerte por carga alta
            else:
                reward = 10 - server_load[server]  # Premiar por baja carga (máximo 10 si carga 0)
            
            # Premiar el equilibrio de tráfico entre servidores
            reward += balance_traffic_reward()
            
            total_reward += reward
            
            # Actualizar tabla Q
            max_q_next = max(Q[router].values())
            Q[router][server] += alpha * (reward + gamma * max_q_next - Q[router][server])
    
    # Almacenar resultados para el análisis
    traffic_history.append(dict(traffic))  # Copiar el estado actual del tráfico
    reward_history.append(total_reward)
    for r in routers:
        q_values_history[r].append(dict(Q[r]))  # Almacenar las tablas Q de cada router
    for s in servidores:
        server_load_history[s].append(server_load[s])
    for r in routers:
        router_traffic_history[r].append(router_traffic[r])

    # Actualizar el timer de los servidores y eliminar solicitudes procesadas
    for s in servidores:
        if server_timers[s] > 0:
            server_timers[s] -= 1  # Reducir el temporizador
        elif server_load[s] > 0:
            server_load[s] -= 1  # Eliminar una solicitud
            server_timers[s] = server_processing_time  # Reiniciar el temporizador para la siguiente solicitud

    # Dibujar nodos con colores y etiquetas de carga
    colors = []
    labels = {}
    for node in G.nodes():
        if node in usuarios:
            colors.append('blue')
            labels[node] = node
        elif node in routers:
            colors.append('green')
            labels[node] = node
        else:  # Servidores
            carga = server_load[node]
            colors.append((1, 0, 0, min(1, carga / 10)))  # Rojo más oscuro si está congestionado
            labels[node] = str(carga)  # Mostrar número de solicitudes en vez del nombre del servidor
    
    # Asignar colores a las aristas en función del tráfico
    edge_colors = []
    for e in edges:
        if traffic[e] > 0:  # Solo colorear si hay tráfico
            edge_colors.append(traffic[e])  # Color según el tráfico
        else:
            edge_colors.append(0)  # Sin tráfico, sin color (invisible)

    # Dibujar el grafo
    nx.draw(G, pos, with_labels=True, labels=labels, node_color=colors, edge_color=edge_colors, node_size=1000, font_size=10)
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_colors, edge_cmap=plt.cm.Reds, width=2)
    
    plt.title(f"Iteración {frame + 1}")

# Función mejorada para elegir el servidor basado en la carga balanceada
def choose_balanced_server(router):
    # Considerar la carga de los servidores de manera más equilibrada
    min_load = min(server_load.values())
    max_load = max(server_load.values())
    
    # Elegir un servidor que esté dentro de un rango balanceado (ni muy vacío ni sobrecargado)
    balanceable_servers = [s for s in servidores if server_load[s] <= (max_load + min_load) / 2]
    if balanceable_servers:
        return random.choice(balanceable_servers)
    
    # Si no hay servidores balanceados, elegir el de menor carga
    return min(server_load, key=server_load.get)

# Función para recompensar el balance de tráfico
def balance_traffic_reward():
    total_traffic = sum(router_traffic.values())
    if total_traffic == 0:
        return 0
    
    # Buscamos que el tráfico sea distribuido de forma más uniforme
    avg_traffic = total_traffic / len(routers)
    reward = 0
    
    for r in routers:
        traffic_diff = abs(router_traffic[r] - avg_traffic)
        reward -= traffic_diff  # Penalizar por desviaciones del promedio
    return reward

# Función para análisis de resultados después de la simulación
def analyze_results():
    # Análisis de la evolución de las tablas Q
    for r in routers:
        plt.figure(figsize=(10, 6))
        q_values = np.array([list(q.values()) for q in q_values_history[r]])
        plt.plot(q_values)
        plt.title(f"Evolución de las Q-values para {r}")
        plt.xlabel("Iteración")
        plt.ylabel("Valor de Q")
        plt.legend(servidores)
        plt.savefig(f"q_values_{r}.png")  # Guardar la gráfica como archivo PNG
        plt.close()  # Cerrar la figura

    # Análisis de la carga de los servidores
    plt.figure(figsize=(10, 6))
    for s in servidores:
        plt.plot(server_load_history[s], label=f"Servidor {s}")
    plt.title("Evolución de la carga de los servidores")
    plt.xlabel("Iteración")
    plt.ylabel("Carga del servidor")
    plt.legend()
    plt.savefig("server_loads.png")  # Guardar la gráfica como archivo PNG
    plt.close()  # Cerrar la figura

    # Análisis de tráfico por router
    plt.figure(figsize=(10, 6))
    for r in routers:
        plt.plot(router_traffic_history[r], label=f"Router {r}")
    plt.title("Evolución del tráfico en los routers")
    plt.xlabel("Iteración")
    plt.ylabel("Tráfico del router")
    plt.legend()
    plt.savefig("router_traffic.png")  # Guardar la gráfica como archivo PNG
    plt.close()  # Cerrar la figura

    # Análisis de recompensas
    plt.figure(figsize=(10, 6))
    plt.plot(reward_history, label="Recompensas acumuladas")
    plt.title("Evolución de las recompensas")
    plt.xlabel("Iteración")
    plt.ylabel("Recompensa")
    plt.legend()
    plt.savefig("rewards.png")  # Guardar la gráfica como archivo PNG
    plt.close()  # Cerrar la figu
    
# Animación
fig = plt.figure(figsize=(8, 6))
ani = animation.FuncAnimation(fig, update, frames=num_iteraciones, repeat=False, interval=500)
plt.show()

# Análisis final
analyze_results()
