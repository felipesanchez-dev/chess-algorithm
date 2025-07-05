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
font_large = pygame.font.Font(None, 36)
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

    def draw(self, WIN, gap):
        color = self.colour
        
        # Cambiar color si está seleccionada o es un movimiento legal
        if self.is_selected:
            color = YELLOW
        elif self.is_legal_move:
            color = LIGHT_GREEN
        
        pygame.draw.rect(WIN, color, (self.x, self.y, gap, gap))
        
    def update_chess_state(self, board):
        """Actualiza el estado del nodo basándose en el tablero de ajedrez"""
        # Convertir coordenadas del nodo a coordenadas de ajedrez
        chess_square = chess.square(self.col, 7-self.row)
        piece = board.piece_at(chess_square)
        
        # Actualizar el estado visual basándose en la pieza
        if piece:
            self.piece = piece
        else:
            self.piece = None

def make_grid(rows, width):
    grid = []
    gap = width // rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            node = Node(i, j, gap)
            if (i + j) % 2 == 0:
                node.colour = WHITE
            else:
                node.colour = GREY
            node.original_colour = node.colour
            grid[i].append(node)
    return grid

def get_clicked_pos(pos, rows, width):
    gap = width // rows
    x, y = pos
    row = y // gap
    col = x // gap
    return row, col

def clear_highlights(grid):
    """Limpia todos los highlights del tablero"""
    for row in grid:
        for node in row:
            node.is_selected = False
            node.is_legal_move = False

def highlight_legal_moves(grid, square):
    """Resalta los movimientos legales para una pieza"""
    legal_moves = list(board.legal_moves)
    
    for move in legal_moves:
        if move.from_square == square:
            to_row = 7 - chess.square_rank(move.to_square)
            to_col = chess.square_file(move.to_square)
            if 0 <= to_row < 8 and 0 <= to_col < 8:
                grid[to_row][to_col].is_legal_move = True

def draw_grid(WIN, grid, rows, width):
    """Dibuja el tablero de ajedrez"""
    gap = width // rows
    for i in range(rows):
        for j in range(rows):
            grid[i][j].draw(WIN, gap)
            
            # Dibujar coordenadas
            if j == 0:  # Números en el lado izquierdo
                rank_text = font_small.render(str(8-i), True, BLACK)
                WIN.blit(rank_text, (5, i * gap + 5))
            if i == 7:  # Letras en la parte inferior
                file_text = font_small.render(chr(ord('a') + j), True, BLACK)
                WIN.blit(file_text, (j * gap + gap - 15, 7 * gap + gap - 20))
    
    # Dibujar las piezas
    for i in range(rows):
        for j in range(rows):
            chess_square = chess.square(j, 7-i)
            piece = board.piece_at(chess_square)
            if piece:
                piece_image = get_piece_image(piece)
                if piece_image:
                    # Escalar la imagen si es necesario
                    piece_size = int(gap * 0.8)  # 80% del tamaño de la casilla
                    piece_image = pygame.transform.scale(piece_image, (piece_size, piece_size))
                    piece_rect = piece_image.get_rect()
                    piece_rect.center = (j * gap + gap // 2, i * gap + gap // 2)
                    WIN.blit(piece_image, piece_rect)
                else:
                    # Si no hay imagen, dibujar texto como alternativa
                    piece_symbol = piece.symbol()
                    # Usar símbolos Unicode para las piezas
                    piece_chars = {
                        'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
                        'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
                    }
                    display_char = piece_chars.get(piece_symbol, piece_symbol)
                    color = BLACK if piece.color == chess.WHITE else (100, 100, 100)
                    piece_text = font_large.render(display_char, True, color)
                    text_rect = piece_text.get_rect()
                    text_rect.center = (j * gap + gap // 2, i * gap + gap // 2)
                    WIN.blit(piece_text, text_rect)

def get_piece_image(piece):
    """Obtiene la imagen de la pieza"""
    piece_files = {
        'P': 'Images/wP.png', 'N': 'Images/wN.png', 'B': 'Images/wB.png',
        'R': 'Images/wR.png', 'Q': 'Images/wQ.png', 'K': 'Images/wK.png',
        'p': 'Images/bP.png', 'n': 'Images/bN.png', 'b': 'Images/bB.png',
        'r': 'Images/bR.png', 'q': 'Images/bQ.png', 'k': 'Images/bK.png',
    }
    
    try:
        if piece.symbol() in piece_files:
            image_path = piece_files[piece.symbol()]
            return pygame.image.load(image_path)
    except Exception as e:
        print(f"⚠️ Error cargando imagen para {piece.symbol()}: {e}")
    
    return None

def getValueOfPiece(piece):
    """Obtiene el valor de una pieza"""
    if piece is None or str(piece) == "None":
        return 0
    
    piece_str = str(piece)
    
    if piece_str == "P":
        return 10
    elif piece_str == "N":
        return 30
    elif piece_str == "B":
        return 30
    elif piece_str == "R":
        return 50
    elif piece_str == "Q":
        return 90
    elif piece_str == "K":
        return 900
    elif piece_str == "p":
        return -10
    elif piece_str == "n":
        return -30
    elif piece_str == "b":
        return -30
    elif piece_str == "r":
        return -50
    elif piece_str == "q":
        return -90
    elif piece_str == "k":
        return -900
    else:
        return 0

def evaluateBoard(boardCopy, movement):
    """Evalúa el tablero"""
    evaluation = 0
    for square in chess.SQUARES:
        piece = boardCopy.piece_at(square)
        evaluation += getValueOfPiece(piece)
    return evaluation

def machine_move(boardCopy):
    legal_moves = list(boardCopy.legal_moves)  # Usar list() directamente
    if not legal_moves:
        return None
    
    best_movement = None
    best_score = -999999
    move_evaluations = []
    
    # Inicializar neural_weight fuera del bucle para evitar errores
    neural_weight = min(0.8, 0.5 + (neural_net.games_played * 0.01))  # Aumenta con experiencia
    
    for move in legal_moves:
        try:
            move_str = move.uci()
            
            # Evaluación híbrida mejorada
            neural_score = neural_evaluate_move(boardCopy, move, neural_net) * 1000
            traditional_score = alphabeta_pruning(boardCopy.copy(), move_str, 2, -999999, 999999, False)
            
            # Bonus por coronación a dama
            if move.promotion == chess.QUEEN:
                traditional_score += 800
                print(f"🤖👑 IA considera coronación a Dama: +800 puntos")
            elif move.promotion:
                traditional_score += 200
                print(f"🤖👑 IA considera coronación: +200 puntos")
            
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
                best_movement = move
            
            # Añadir datos de entrenamiento con análisis detallado
            neural_net.add_game_data(boardCopy, move_str, neural_score, traditional_score)
            
        except Exception as e:
            print(f"Error evaluando movimiento {move}: {e}")
            continue
    
    # Marcar el mejor movimiento en los datos de entrenamiento
    if best_movement and neural_net.training_data:
        best_move_str = best_movement.uci()
        for data in neural_net.training_data:
            if data['move'] == best_move_str:
                data['analysis']['was_best_move'] = True
                break
    
    move_str = best_movement.uci() if best_movement else None
    print(f"🤖 Análisis de movimientos - Neural weight: {neural_weight:.2f}, Best: {move_str} ({best_score:.1f})")
    
    return move_str if best_movement else None

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
    global board, game_moves, game_active, game_start_time
    board = chess.Board()
    game_moves = []
    game_active = True
    game_start_time = pygame.time.get_ticks()
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

# Variables globales adicionales
auto_play_enabled = False
game_end_timer = 0
GAME_END_DELAY = 3000  # 3 segundos antes de iniciar nueva partida

# Crear botones
buttons = create_buttons()

def show_promotion_menu(WIN, pos):
    """Muestra un menú para seleccionar la pieza de coronación"""
    promotion_pieces = [
        (chess.QUEEN, '♕', 'Dama'),
        (chess.ROOK, '♖', 'Torre'),
        (chess.BISHOP, '♗', 'Alfil'),
        (chess.KNIGHT, '♘', 'Caballo')
    ]
    
    # Crear rectángulo para el menú
    menu_width = 200
    menu_height = 150
    menu_x = pos[0] - menu_width // 2
    menu_y = pos[1] - menu_height // 2
    
    # Asegurar que el menú esté dentro de los límites
    menu_x = max(0, min(menu_x, TOTAL_WIDTH - menu_width))
    menu_y = max(0, min(menu_y, WIDTH - menu_height))
    
    menu_rect = pygame.Rect(menu_x, menu_y, menu_width, menu_height)
    
    # Dibujar fondo del menú
    pygame.draw.rect(WIN, LIGHT_BLUE, menu_rect)
    pygame.draw.rect(WIN, BLACK, menu_rect, 3)
    
    # Título
    title_text = font_medium.render("Coronar peón:", True, BLACK)
    WIN.blit(title_text, (menu_x + 10, menu_y + 10))
    
    # Opciones
    button_height = 25
    button_width = 180
    buttons = []
    
    for i, (piece_type, symbol, name) in enumerate(promotion_pieces):
        button_y = menu_y + 40 + i * (button_height + 5)
        button_rect = pygame.Rect(menu_x + 10, button_y, button_width, button_height)
        
        # Dibujar botón
        pygame.draw.rect(WIN, WHITE, button_rect)
        pygame.draw.rect(WIN, BLACK, button_rect, 2)
        
        # Texto del botón
        button_text = font_small.render(f"{symbol} {name}", True, BLACK)
        text_rect = button_text.get_rect(center=button_rect.center)
        WIN.blit(button_text, text_rect)
        
        buttons.append((button_rect, piece_type))
    
    return buttons

def handle_promotion_selection(WIN, pos, selected_square, target_square, board):
    """Maneja la selección de coronación"""
    buttons = show_promotion_menu(WIN, pos)
    pygame.display.update()
    
    waiting_for_selection = True
    selected_piece = chess.QUEEN  # Por defecto
    
    while waiting_for_selection:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                
                # Verificar si se hizo clic en algún botón
                for button_rect, piece_type in buttons:
                    if button_rect.collidepoint(mouse_pos):
                        selected_piece = piece_type
                        waiting_for_selection = False
                        break
                
                # Si se hizo clic fuera del menú, usar dama por defecto
                if waiting_for_selection:
                    in_menu = False
                    for button_rect, _ in buttons:
                        if button_rect.collidepoint(mouse_pos):
                            in_menu = True
                            break
                    if not in_menu:
                        selected_piece = chess.QUEEN
                        waiting_for_selection = False
            
            if event.type == pygame.KEYDOWN:
                # Teclas rápidas para coronación
                if event.key == pygame.K_q:
                    selected_piece = chess.QUEEN
                    waiting_for_selection = False
                elif event.key == pygame.K_r:
                    selected_piece = chess.ROOK
                    waiting_for_selection = False
                elif event.key == pygame.K_b:
                    selected_piece = chess.BISHOP
                    waiting_for_selection = False
                elif event.key == pygame.K_n:
                    selected_piece = chess.KNIGHT
                    waiting_for_selection = False
                elif event.key == pygame.K_ESCAPE:
                    selected_piece = chess.QUEEN
                    waiting_for_selection = False
    
    return selected_piece

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
        pygame.time.delay(50)
        current_time = pygame.time.get_ticks()
        
        for event in pygame.event.get():
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
                if event.key == pygame.K_r:
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
                
                print(f"🖱️ Clic en posición: {pos}, fila: {row}, columna: {col}")
                
                if 0 <= row < 8 and 0 <= col < 8:
                    square = chess.square(col, 7-row)
                    piece = board.piece_at(square)
                    
                    print(f"🎯 Casilla válida: {chess.square_name(square)}, Pieza: {piece}")
                    
                    if selected_square is None:
                        # Seleccionar pieza
                        if piece and piece.color == chess.WHITE and board.turn == chess.WHITE:
                            selected_square = square
                            clear_highlights(grid)
                            grid[row][col].is_selected = True
                            highlight_legal_moves(grid, square)
                            print(f"✅ Pieza seleccionada: {piece.symbol()} en {chess.square_name(square)}")
                        else:
                            print("❌ No hay pieza blanca válida en esa casilla")
                    else:
                        # Intentar movimiento
                        move = chess.Move(selected_square, square)
                        
                        # Verificar si es un movimiento de coronación
                        piece_moving = board.piece_at(selected_square)
                        is_promotion = False
                        
                        if (piece_moving and piece_moving.piece_type == chess.PAWN and 
                            ((piece_moving.color == chess.WHITE and chess.square_rank(square) == 7) or
                             (piece_moving.color == chess.BLACK and chess.square_rank(square) == 0))):
                            # Es coronación, mostrar menú de selección
                            is_promotion = True
                            print(f"👑 Coronación de peón detectada!")
                            
                            # Mostrar menú de coronación
                            selected_piece = handle_promotion_selection(WIN, pos, selected_square, square, board)
                            move = chess.Move(selected_square, square, promotion=selected_piece)
                            
                            piece_names = {
                                chess.QUEEN: "Dama",
                                chess.ROOK: "Torre", 
                                chess.BISHOP: "Alfil",
                                chess.KNIGHT: "Caballo"
                            }
                            print(f"👑 Coronación a {piece_names[selected_piece]}!")
                        
                        # Verificar si el movimiento es legal
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
                                print(f"🔄 Cambiando selección a: {piece.symbol()} en {chess.square_name(square)}")
                            else:
                                print("❌ Movimiento inválido")
                                selected_square = None
                                clear_highlights(grid)
                else:
                    print(f"❌ Clic fuera del tablero: fila {row}, columna {col}")
        
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

if __name__ == "__main__":
    main(WIN, WIDTH)
