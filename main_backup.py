import random
import pygame
import sys
import chess
from chess_neural_network import AdvancedChessNeuralNetwork, neural_evaluate_move

board = chess.Board()
neural_net = AdvancedChessNeuralNetwork()

# Variables para el seguimiento de la partida
game_moves = []
game_active = True
game_start_time = None

WIDTH = 650
PANEL_WIDTH = 300
TOTAL_WIDTH = WIDTH + PANEL_WIDTH

pygame.init()
WIN = pygame.display.set_mode((TOTAL_WIDTH, WIDTH))
pygame.display.set_caption("Ajedrez con IA Neural - Aprendizaje Automático")
WHITE = (255, 255, 255)
GREY = (128, 128, 128)
YELLOW = (204, 204, 0)
BLUE = (50, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)  # Para movimientos legales
RED = (255, 0, 0)    # Para pieza seleccionada
LIGHT_GREEN = (144, 238, 144)  # Para movimientos legales más suave
DARK_BLUE = (25, 25, 112)
LIGHT_BLUE = (173, 216, 230)
ORANGE = (255, 165, 0)

# Fuentes
font_large = pygame.font.Font(None, 24)
font_medium = pygame.font.Font(None, 20)
font_small = pygame.font.Font(None, 16)

class Node:
    def __init__(self, row, col, width):
        self.row = row
        self.col = col
        self.x = int(col * width)
        self.y = int(row * width)
        self.colour = WHITE
        self.is_selected = False
        self.is_legal_move = False
        self.original_colour = WHITE

    def draw(self, WIN):
        color = self.colour
        
        # Cambiar color si está seleccionada o es un movimiento legal
        if self.is_selected:
            color = YELLOW
        elif self.is_legal_move:
            color = LIGHT_GREEN
        
        pygame.draw.rect(WIN, color, (self.x, self.y, WIDTH / 8, WIDTH / 8))
        
        # Dibujar un círculo en el centro para movimientos legales
        if self.is_legal_move and not self.is_selected:
            center_x = self.x + WIDTH // 16
            center_y = self.y + WIDTH // 16
            pygame.draw.circle(WIN, GREEN, (center_x, center_y), 15, 3)

    def setup(self, WIN, boardM):
        if self.row < len(boardM) and self.col < len(boardM[self.row]):
            piece = boardM[self.row][self.col]
            if piece != "None":
                try:
                    image_path = self.getImage(piece)
                    if image_path:
                        # Verificar si el archivo existe
                        import os
                        if os.path.exists(image_path):
                            image = pygame.image.load(image_path)
                            # Redimensionar la imagen para que se ajuste al tamaño de la casilla
                            image = pygame.transform.scale(image, (int(WIDTH / 8), int(WIDTH / 8)))
                            WIN.blit(image, (self.x, self.y))
                except:
                    pass  # Silenciar errores para mejor rendimiento
    
    def getImage(self, letter):
        if letter == 'r':
            return "Images/bR.png"
        if letter == 'n':
            return "Images/bN.png"
        if letter == 'b':
            return "Images/bB.png"
        if letter == 'q':
            return "Images/bQ.png"
        if letter == 'k':
            return "Images/bK.png"
        if letter == 'p':
            return "Images/bP.png"
        
        if letter == 'R':
            return "Images/wR.png"
        if letter == 'N':
            return "Images/wN.png"
        if letter == 'B':
            return "Images/wB.png"
        if letter == 'Q':
            return "Images/wQ.png"
        if letter == 'K':
            return "Images/wK.png"
        if letter == 'P':
            return "Images/wP.png"

        return ""


def make_grid(rows, width):
    grid = []
    gap = WIDTH // rows
    print(gap)
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            node = Node(i, j, gap)
            grid[i].append(node)
            if (i+j)%2 ==1:
                node.colour = GREY
                node.original_colour = GREY
            else:
                node.original_colour = WHITE
    return grid

def clear_highlights(grid):
    """Limpia todos los resaltados del tablero"""
    for row in grid:
        for spot in row:
            spot.is_selected = False
            spot.is_legal_move = False

def highlight_legal_moves(grid, selected_square):
    """Resalta los movimientos legales para la pieza seleccionada"""
    clear_highlights(grid)
    
    # Resaltar la casilla seleccionada
    selected_pos = chess.parse_square(selected_square)
    selected_file = chess.square_file(selected_pos)
    selected_rank = chess.square_rank(selected_pos)
    
    # Convertir a coordenadas de grid
    grid_row = 7 - selected_rank
    grid_col = selected_file
    
    if 0 <= grid_row < 8 and 0 <= grid_col < 8:
        grid[grid_row][grid_col].is_selected = True
    
    # Resaltar movimientos legales
    legal_moves = list(board.legal_moves)
    for move in legal_moves:
        if move.from_square == selected_pos:
            to_file = chess.square_file(move.to_square)
            to_rank = chess.square_rank(move.to_square)
            
            # Convertir a coordenadas de grid
            grid_row = 7 - to_rank
            grid_col = to_file
            
            if 0 <= grid_row < 8 and 0 <= grid_col < 8:
                grid[grid_row][grid_col].is_legal_move = True

def is_legal_move_target(square, selected_square):
    """Verifica si un movimiento es legal"""
    if not selected_square:
        return False
    
    try:
        move = chess.Move.from_uci(selected_square + square)
        return move in board.legal_moves
    except:
        return False

def draw_grid(win, rows, width):
    gap = width // 8
    for i in range(rows):
        pygame.draw.line(win, BLACK, (0, i * gap), (width, i * gap))
        for j in range(rows):
            pygame.draw.line(win, BLACK, (j * gap, 0), (j * gap, width))

def update_display(win, grid, rows, width):
    win.fill(WHITE)  # Limpiar la pantalla primero
    
    boardM = []
    for row in range(8):
        arr = []
        for col in range(8):
            piece = board.piece_at(chess.square(col, 7-row))  # 7-row para invertir el orden
            if piece is None:
                arr.append("None")
            else:
                arr.append(str(piece))
        boardM.append(arr)

    # Primero dibujar las casillas del tablero
    for row in grid:
        for spot in row:
            spot.draw(win)
    
    # Luego dibujar las líneas de la cuadrícula
    draw_grid(win, rows, width)
    
    # Finalmente dibujar las fichas encima
    for row in grid:
        for spot in row:
            spot.setup(win, boardM)
    
    pygame.display.update()

def Find_Node(pos, WIDTH):
    interval = WIDTH / 8
    x, y = pos
    col = int(x // interval)
    row = int(y // interval)
    
    # Asegurar que las coordenadas estén dentro del rango válido
    if 0 <= col < 8 and 0 <= row < 8:
        pos = ['a','b','c','d','e','f','g','h'][col] + str(8-row)
        return pos
    return None

def machine_move(boardCopy):
    legal_moves = [str(mov) for mov in boardCopy.legal_moves]
    if not legal_moves:
        return None
    
    best_movement = ""
    best_score = -999999
    move_evaluations = []
    
    # Inicializar neural_weight fuera del bucle para evitar errores
    neural_weight = min(0.8, 0.5 + (neural_net.games_played * 0.01))  # Aumenta con experiencia
    
    for move_str in legal_moves:
        try:
            move = chess.Move.from_uci(move_str)
            
            # Evaluación híbrida mejorada
            neural_score = neural_evaluate_move(boardCopy, move, neural_net) * 1000
            traditional_score = alphabeta_pruning(boardCopy.copy(), move_str, 2, -999999, 999999, False)
            
            # Combinar puntuaciones con peso adaptativo
            traditional_weight = 1 - neural_weight
            
            combined_score = neural_weight * neural_score + traditional_weight * traditional_score
            
            move_evaluations.append({
                'move': move_str,
                'neural': neural_score,
                'traditional': traditional_score,
                'combined': combined_score
            })
            
            if combined_score > best_score:
                best_score = combined_score
                best_movement = move_str
            
            # Añadir datos de entrenamiento con análisis detallado
            neural_net.add_game_data(boardCopy, move_str, neural_score, traditional_score)
            
        except Exception as e:
            print(f"Error evaluando movimiento {move_str}: {e}")
            continue
    
    # Marcar el mejor movimiento en los datos de entrenamiento
    if best_movement and neural_net.training_data:
        for data in neural_net.training_data:
            if data['move'] == best_movement:
                data['analysis']['was_best_move'] = True
                break
    
    print(f"🤖 Análisis de movimientos - Neural weight: {neural_weight:.2f}, Best: {best_movement} ({best_score:.1f})")
    
    return best_movement if best_movement else legal_moves[0]

def alphabeta_pruning(boardCopy,movement,depth,alpha,beta,maximizingPlayer):
    if depth == 0:
        return evaluateBoard(boardCopy,movement)
    
    boardCopy.push(chess.Move.from_uci(movement))
    legal_moves = [str(mov) for mov in boardCopy.legal_moves]

    if maximizingPlayer:
        value = -999999
        for move in legal_moves:
            value = max(value,alphabeta_pruning(boardCopy.copy(),move,depth-1,alpha,beta,False))
            if value >= beta:
                break
            alpha = max(alpha,value)
        return value
    else:
        value = 999999
        for move in legal_moves:
            value = min(value,alphabeta_pruning(boardCopy.copy(),move,depth-1,alpha,beta,True))
            if value <= alpha:
                break
            beta = min(beta,value)
        return value

def evaluateBoard(boardCopy,movement):
    value = 0
    boardCopy.push(chess.Move.from_uci(movement))
    for i in range(8):
        for j in range(8):
            piece = str(boardCopy.piece_at(chess.Square((i*8+j))))
            value += getValueOfPiece(piece)
    return value

def getValueOfPiece(letter):
        if letter == 'r':
            return 50
        if letter == 'n':
            return 30
        if letter == 'b':
            return 30
        if letter == 'q':
            return 90
        if letter == 'k':
            return 900
        if letter == 'p':
            return 10
        
        if letter == 'R':
            return -50
        if letter == 'N':
            return -30
        if letter == 'B':
            return -30
        if letter == 'Q':
            return -90
        if letter == 'K':
            return -900
        if letter == 'P':
            return -10
        
        # Retornar 0 para casillas vacías o piezas no reconocidas
        return 0

def check_game_end(board):
    """Verifica si el juego ha terminado y devuelve el resultado"""
    if board.is_checkmate():
        if board.turn == chess.WHITE:
            return -1  # Máquina (negras) gana
        else:
            return 1   # Jugador (blancas) gana
    elif board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
        return 0  # Empate
    return None  # Juego continúa

def end_game_and_train(result):
    """Termina el juego y entrena la red neuronal"""
    global game_active, board, neural_net, game_start_time
    
    game_duration = (pygame.time.get_ticks() - game_start_time) / 1000 if game_start_time else 0
    
    if result == 1:
        print("🎉 ¡Felicidades! Has ganado la partida!")
        neural_net.train_from_game(-1)  # La máquina pierde
    elif result == -1:
        print("🤖 La máquina ha ganado la partida.")
        neural_net.train_from_game(1)   # La máquina gana
    else:
        print("🤝 La partida terminó en empate.")
        neural_net.train_from_game(0)   # Empate
    
    # Mostrar estadísticas actualizadas
    stats = neural_net.get_stats()
    print(f"\n🧠 === ESTADÍSTICAS DE LA RED NEURONAL ===")
    print(f"🎮 Partidas jugadas: {stats['games_played']}")
    print(f"🏆 Victorias de la máquina: {stats['wins']}")
    print(f"😞 Derrotas de la máquina: {stats['losses']}")
    print(f"🤝 Empates: {stats['draws']}")
    print(f"📊 Tasa de victoria: {stats['win_rate']:.1f}%")
    print(f"🎯 Precisión del modelo: {stats['accuracy']:.1f}%")
    print(f"🧠 Tasa de aprendizaje: {stats['learning_rate']:.6f}")
    print(f"💾 Experiencias almacenadas: {stats['experience_replay_size']}")
    print(f"⏱️ Duración de la partida: {game_duration:.1f}s")
    print(f"♟️ Movimientos totales: {len(game_moves)}")
    print("=" * 45)
    
    # Mostrar evolución del aprendizaje
    if stats['games_played'] > 1:
        if stats['games_played'] <= 5:
            print("🌱 La red neuronal está en fase de aprendizaje inicial")
        elif stats['games_played'] <= 20:
            print("🌿 La red neuronal está desarrollando patrones básicos")
        elif stats['games_played'] <= 50:
            print("🌳 La red neuronal está mejorando su comprensión táctica")
        else:
            print("🧠 La red neuronal ha desarrollado conocimiento avanzado")
    
    print("\n📋 Controles:")
    print("   R: Reiniciar nueva partida")
    print("   Q: Salir del juego")
    print("   A: Mostrar análisis de progreso")
    print("   S: Guardar modelo manualmente")
    
    game_active = False

class Button:
    def __init__(self, x, y, width, height, text, color, text_color, action=None):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.text_color = text_color
        self.action = action
        self.hovered = False
        self.original_color = color
        self.hover_color = (min(255, color[0] + 30), min(255, color[1] + 30), min(255, color[2] + 30))
    
    def draw(self, surface):
        # Cambiar color si está siendo hover
        current_color = self.hover_color if self.hovered else self.original_color
        pygame.draw.rect(surface, current_color, self.rect)
        pygame.draw.rect(surface, BLACK, self.rect, 2)
        
        # Texto centrado
        text_surface = font_medium.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)
    
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                if self.action:
                    self.action()
                return True
        elif event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        return False

def reset_game():
    """Reinicia el juego"""
    global board, game_moves, game_active, game_start_time, selected_piece, legal_moves
    board = chess.Board()
    game_moves = []
    game_active = True
    game_start_time = pygame.time.get_ticks()
    selected_piece = None
    legal_moves = []
    print("🔄 Nueva partida iniciada")

def save_model():
    """Guarda el modelo manualmente"""
    neural_net.save_model()
    print("💾 Modelo guardado manualmente")

def show_analysis():
    """Muestra análisis de progreso"""
    neural_net.analyze_learning_progress()
    print("📊 Análisis mostrado en consola")

def auto_play_mode():
    """Modo de juego automático"""
    global auto_play_enabled
    auto_play_enabled = not auto_play_enabled
    status = "activado" if auto_play_enabled else "desactivado"
    print(f"🤖 Modo automático {status}")

def draw_control_panel():
    """Dibuja el panel de control lateral"""
    panel_x = WIDTH
    panel_rect = pygame.Rect(panel_x, 0, PANEL_WIDTH, WIDTH)
    pygame.draw.rect(WIN, LIGHT_BLUE, panel_rect)
    pygame.draw.rect(WIN, BLACK, panel_rect, 2)
    
    # Título
    title_text = font_large.render("Control Panel", True, DARK_BLUE)
    WIN.blit(title_text, (panel_x + 10, 10))
    
    # Estadísticas del juego
    stats = neural_net.get_stats()
    y_offset = 50
    
    stats_info = [
        f"🎮 Partidas: {stats['games_played']}",
        f"🏆 Victorias: {stats['wins']}",
        f"😞 Derrotas: {stats['losses']}",
        f"🤝 Empates: {stats['draws']}",
        f"📊 Tasa victoria: {stats['win_rate']:.1f}%",
        f"🎯 Precisión: {stats['accuracy']:.1f}%",
        f"🧠 Tasa aprend.: {stats['learning_rate']:.6f}",
        f"💾 Experiencias: {stats['experience_replay_size']}"
    ]
    
    for i, info in enumerate(stats_info):
        text = font_small.render(info, True, BLACK)
        WIN.blit(text, (panel_x + 10, y_offset + i * 20))
    
    # Estado del juego actual
    y_offset += len(stats_info) * 20 + 20
    
    if board.is_checkmate():
        result_text = "🏁 JAQUE MATE"
        if board.turn == chess.WHITE:
            result_text += " - IA GANA!"
        else:
            result_text += " - JUGADOR GANA!"
        color = RED
    elif board.is_stalemate():
        result_text = "🤝 EMPATE"
        color = ORANGE
    elif board.is_check():
        result_text = "⚠️ JAQUE"
        color = ORANGE
    else:
        turn_text = "Turno: " + ("Jugador" if board.turn == chess.WHITE else "IA")
        result_text = turn_text
        color = BLACK
    
    game_state_text = font_medium.render(result_text, True, color)
    WIN.blit(game_state_text, (panel_x + 10, y_offset))
    
    # Información de la partida actual
    y_offset += 40
    if game_start_time:
        elapsed_time = (pygame.time.get_ticks() - game_start_time) / 1000
        time_text = f"⏱️ Tiempo: {elapsed_time:.1f}s"
        moves_text = f"♟️ Movimientos: {len(game_moves)}"
        
        time_surface = font_small.render(time_text, True, BLACK)
        moves_surface = font_small.render(moves_text, True, BLACK)
        WIN.blit(time_surface, (panel_x + 10, y_offset))
        WIN.blit(moves_surface, (panel_x + 10, y_offset + 20))

def create_buttons():
    """Crea los botones del panel de control"""
    panel_x = WIDTH
    buttons = []
    
    # Botón Nueva Partida
    buttons.append(Button(panel_x + 10, 400, 120, 35, "Nueva Partida", GREEN, BLACK, reset_game))
    
    # Botón Guardar Modelo
    buttons.append(Button(panel_x + 140, 400, 120, 35, "Guardar Modelo", BLUE, WHITE, save_model))
    
    # Botón Análisis
    buttons.append(Button(panel_x + 10, 445, 120, 35, "Ver Análisis", YELLOW, BLACK, show_analysis))
    
    # Botón Modo Auto
    buttons.append(Button(panel_x + 140, 445, 120, 35, "Modo Auto", ORANGE, BLACK, auto_play_mode))
    
    # Botón Salir
    buttons.append(Button(panel_x + 75, 490, 120, 35, "Salir", RED, WHITE, lambda: pygame.quit()))
    
    return buttons

# Variables globales adicionales
auto_play_enabled = False
game_end_timer = 0
GAME_END_DELAY = 3000  # 3 segundos antes de iniciar nueva partida

# Crear botones
buttons = create_buttons()

def main(WIN, WIDTH):
    global game_active, board, game_start_time, auto_play_enabled, game_end_timer
    selected_square = None
    grid = make_grid(8, WIDTH)
    
    # Mostrar estadísticas iniciales
    stats = neural_net.get_stats()
    print("🧠 === AJEDREZ CON RED NEURONAL AVANZADA ===")
    print(f"🎮 Partidas previas: {stats['games_played']}")
    print(f"🏆 Tasa de victoria: {stats['win_rate']:.1f}%")
    print(f"📊 Precisión: {stats['accuracy']:.1f}%")
    print(f"🧠 Tasa de aprendizaje: {stats['learning_rate']:.6f}")
    print(f"💾 Experiencias almacenadas: {stats['experience_replay_size']}")
    
    # Analizar progreso si hay datos previos
    if stats['games_played'] > 0:
        neural_net.analyze_learning_progress()
    
    print("=" * 50)
    
    # Debug: mostrar el estado inicial del tablero
    print("Estado inicial del tablero:")
    print(board)
    
    game_start_time = pygame.time.get_ticks()
    
    while True:
        pygame.time.delay(50) ##stops cpu dying
        current_time = pygame.time.get_ticks()
        
        for event in pygame.event.get(): #This quits the program if the player closes the window
            if event.type == pygame.QUIT:
                # Exportar resumen antes de salir
                neural_net.export_training_summary()
                pygame.quit()
                sys.exit()
            
            # Manejar eventos de botones
            for button in buttons:
                if button.handle_event(event):
                    if button.text == "Salir":
                        neural_net.export_training_summary()
                        pygame.quit()
                        sys.exit()
                    break
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r and not game_active:
                    reset_game()
                    selected_square = None
                    clear_highlights(grid)
                    game_start_time = pygame.time.get_ticks()
                elif event.key == pygame.K_q:
                    neural_net.export_training_summary()
                    pygame.quit()
                    sys.exit()
                elif event.key == pygame.K_a:
                    show_analysis()
                elif event.key == pygame.K_s:
                    save_model()
                elif event.key == pygame.K_m:
                    auto_play_mode()
            
            # Manejo de clics del mouse para el tablero
            if event.type == pygame.MOUSEBUTTONDOWN and event.pos[0] < WIDTH:
                if not game_active:
                    continue
                    
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, 8, WIDTH)
                
                if 0 <= row < 8 and 0 <= col < 8:
                    square = chess.square(col, 7-row)
                    piece = board.piece_at(square)
                    
                    if selected_square is None:
                        # Seleccionar pieza
                        if piece and piece.color == chess.WHITE and board.turn == chess.WHITE:
                            selected_square = square
                            clear_highlights(grid)
                            grid[row][col].is_selected = True
                            highlight_legal_moves(grid, square)
                            print(f"Pieza seleccionada: {piece.symbol()} en {chess.square_name(square)}")
                        else:
                            print("No hay pieza blanca válida en esa casilla")
                    else:
                        # Intentar movimiento
                        move = chess.Move(selected_square, square)
                        if move in board.legal_moves:
                            # Realizar movimiento
                            board.push(move)
                            game_moves.append(move.uci())
                            print(f"✅ Movimiento realizado: {move.uci()}")
                            
                            # Limpiar selección
                            selected_square = None
                            clear_highlights(grid)
                            
                            # Verificar fin del juego
                            if check_game_end():
                                continue
                            
                            # Turno de la máquina
                            if board.turn == chess.BLACK:
                                print("🤖 Turno de la máquina...")
                                machine_movement = machine_move(board.copy())
                                if machine_movement:
                                    board.push(chess.Move.from_uci(machine_movement))
                                    game_moves.append(machine_movement)
                                    print(f"🤖 Movimiento de la máquina: {machine_movement}")
                                    
                                    # Verificar fin del juego después del movimiento de la máquina
                                    check_game_end()
                        else:
                            # Movimiento inválido, cambiar selección si es pieza propia
                            if piece and piece.color == chess.WHITE and board.turn == chess.WHITE:
                                selected_square = square
                                clear_highlights(grid)
                                grid[row][col].is_selected = True
                                highlight_legal_moves(grid, square)
                                print(f"Pieza seleccionada: {piece.symbol()} en {chess.square_name(square)}")
                            else:
                                print("❌ Movimiento inválido")
                                selected_square = None
                                clear_highlights(grid)
        
        # Lógica de juego automático
        if auto_play_enabled and not game_active and game_end_timer > 0:
            if current_time - game_end_timer >= GAME_END_DELAY:
                reset_game()
                selected_square = None
                clear_highlights(grid)
                game_start_time = pygame.time.get_ticks()
                game_end_timer = 0
        
        # Actualizar tablero
        for row in grid:
            for spot in row:
                spot.update_chess_state(board)
        
        # Dibujar todo
        draw_grid(WIN, grid, 8, WIDTH)
        draw_control_panel()
        
        # Dibujar botones
        for button in buttons:
            button.draw(WIN)
        
        pygame.display.update()

def check_game_end():
    """Verifica si el juego ha terminado y maneja el final"""
    global game_active, game_end_timer
    
    if board.is_checkmate():
        game_active = False
        game_end_timer = pygame.time.get_ticks()
        
        if board.turn == chess.WHITE:
            # IA gana
            result = -1
            print("🏁 ¡JAQUE MATE! La IA ha ganado.")
        else:
            # Jugador gana
            result = 1
            print("🏁 ¡JAQUE MATE! El jugador ha ganado.")
        
        # Entrenar la red neuronal
        neural_net.train_from_game(result)
        
        # Mostrar estadísticas finales
        show_game_stats()
        
        return True
    
    elif board.is_stalemate() or board.is_insufficient_material():
        game_active = False
        game_end_timer = pygame.time.get_ticks()
        
        print("🤝 ¡EMPATE!")
        
        # Entrenar con empate
        neural_net.train_from_game(0)
        
        # Mostrar estadísticas finales
        show_game_stats()
        
        return True
    
    return False

def show_game_stats():
    """Muestra las estadísticas del juego"""
    stats = neural_net.get_stats()
    elapsed_time = (pygame.time.get_ticks() - game_start_time) / 1000
    
    print("\n" + "=" * 45)
    print("📊 ESTADÍSTICAS DE LA PARTIDA")
    print("=" * 45)
    print(f"🏆 Victorias de la IA: {stats['wins']}")
    print(f"😞 Derrotas de la máquina: {stats['losses']}")
    print(f"🤝 Empates: {stats['draws']}")
    print(f"📊 Tasa de victoria: {stats['win_rate']:.1f}%")
    print(f"🎯 Precisión del modelo: {stats['accuracy']:.1f}%")
    print(f"🧠 Tasa de aprendizaje: {stats['learning_rate']:.6f}")
    print(f"💾 Experiencias almacenadas: {stats['experience_replay_size']}")
    print(f"⏱️ Duración de la partida: {elapsed_time:.1f}s")
    print(f"♟️ Movimientos totales: {len(game_moves)}")
    print("=" * 45)
    
    if auto_play_enabled:
        print("🤖 Modo automático activado - Nueva partida en 3 segundos...")
    else:
        print("📋 Controles:")
        print("   R: Reiniciar nueva partida")
        print("   M: Activar/desactivar modo automático")
        print("   A: Mostrar análisis de progreso")
        print("   S: Guardar modelo manualmente")
        print("   Q: Salir del juego")
                elif event.key == pygame.K_q and not game_active:
                    neural_net.export_training_summary()
                    pygame.quit()
                    sys.exit()
                elif event.key == pygame.K_s:  # Guardar progreso manualmente
                    neural_net.save_model()
                    print("💾 Modelo guardado manualmente")
                elif event.key == pygame.K_a and not game_active:  # Análisis
                    neural_net.analyze_learning_progress()

            if event.type == pygame.MOUSEBUTTONDOWN and game_active:
                node = Find_Node(pygame.mouse.get_pos(), WIDTH)
                if node is None:  # Clic fuera del tablero
                    continue
                    
                if selected_square is None:
                    # Primer clic: seleccionar pieza
                    piece = board.piece_at(chess.parse_square(node))
                    if piece != None and str(piece).isupper(): 
                        selected_square = node
                        highlight_legal_moves(grid, selected_square)
                        print(f"Pieza seleccionada: {piece} en {node}")
                    else:
                        print("No hay pieza blanca válida en esa casilla")
                else:
                    # Segundo clic: mover pieza o cambiar selección
                    if node == selected_square:
                        # Clic en la misma casilla: deseleccionar
                        print("Pieza deseleccionada")
                        selected_square = None
                        clear_highlights(grid)
                    elif is_legal_move_target(node, selected_square):
                        # Movimiento legal: ejecutar inmediatamente
                        full_movement = selected_square + node
                        try:
                            move = chess.Move.from_uci(full_movement)
                            board.push(move)
                            game_moves.append(full_movement)
                            print(f"✅ Movimiento realizado: {full_movement}")
                            selected_square = None
                            clear_highlights(grid)
                            
                            # Verificar si el juego terminó después del movimiento del jugador
                            result = check_game_end(board)
                            if result is not None:
                                end_game_and_train(result)
                                continue
                            
                            # Turno de la máquina
                            print("🤖 Turno de la máquina...")
                            machine_movement = machine_move(board.copy())
                            if machine_movement:
                                board.push(chess.Move.from_uci(machine_movement))
                                game_moves.append(machine_movement)
                                print(f"🤖 Movimiento de la máquina: {machine_movement}")
                                
                                # Verificar si el juego terminó después del movimiento de la máquina
                                result = check_game_end(board)
                                if result is not None:
                                    end_game_and_train(result)
                                    continue
                                    
                        except Exception as e:
                            print(f"❌ Error ejecutando movimiento: {e}")
                            selected_square = None
                            clear_highlights(grid)
                    else:
                        # Clic en otra pieza propia: cambiar selección
                        piece = board.piece_at(chess.parse_square(node))
                        if piece != None and str(piece).isupper():
                            selected_square = node
                            highlight_legal_moves(grid, selected_square)
                            print(f"Nueva pieza seleccionada: {piece} en {node}")
                        else:
                            print("Movimiento no válido")
                            selected_square = None
                            clear_highlights(grid)

        # Actualizar la pantalla en cada iteración del bucle principal
        update_display(WIN, grid, 8, WIDTH)
        
        # Mostrar mensaje de fin de juego
        if not game_active:
            # Mostrar controles
            font = pygame.font.Font(None, 36)
            text = font.render("R: Nueva partida | Q: Salir | A: Análisis", True, BLACK)
            WIN.blit(text, (10, 10))
        
        # Dibujar botones
        for button in buttons:
            button.draw(WIN)
        
        # Manejar eventos de botones
        for event in pygame.event.get():
            for button in buttons:
                button.handle_event(event)

main(WIN, WIDTH)
