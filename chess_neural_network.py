import numpy as np
import chess
import pickle
import os
import json
import csv
import pandas as pd
from datetime import datetime
import random

class AdvancedChessNeuralNetwork:
    def __init__(self, input_size=768, hidden_layers=[512, 256, 128], output_size=1, learning_rate=0.001):
        """
        Red neuronal avanzada para evaluación de posiciones de ajedrez
        input_size: 768 = 8x8x12 (64 casillas x 12 tipos de piezas)
        hidden_layers: capas ocultas con diferentes tamaños
        output_size: 1 (evaluación de la posición)
        """
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Inicializar pesos y bias para múltiples capas
        self.weights = []
        self.biases = []
        
        # Primera capa
        self.weights.append(np.random.randn(input_size, hidden_layers[0]) * np.sqrt(2.0 / input_size))
        self.biases.append(np.zeros((1, hidden_layers[0])))
        
        # Capas ocultas
        for i in range(1, len(hidden_layers)):
            self.weights.append(np.random.randn(hidden_layers[i-1], hidden_layers[i]) * np.sqrt(2.0 / hidden_layers[i-1]))
            self.biases.append(np.zeros((1, hidden_layers[i])))
        
        # Capa de salida
        self.weights.append(np.random.randn(hidden_layers[-1], output_size) * np.sqrt(2.0 / hidden_layers[-1]))
        self.biases.append(np.zeros((1, output_size)))
        
        # Historial de partidas y análisis
        self.game_history = []
        self.training_data = []
        self.position_evaluations = []
        self.move_analysis = []
        
        # Estadísticas avanzadas
        self.games_played = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.total_moves = 0
        self.correct_predictions = 0
        self.prediction_accuracy = 0.0
        
        # Memoria de experiencias (para replay)
        self.experience_replay = []
        self.max_replay_size = 10000
        
        # Parámetros de aprendizaje adaptativos
        self.initial_learning_rate = learning_rate
        self.learning_decay = 0.995
        self.min_learning_rate = 0.0001
        
        # Crear directorio para datos
        os.makedirs('neural_data', exist_ok=True)
        
        # Cargar modelo si existe
        self.load_model()
        
        # Inicializar archivos CSV
        self.setup_csv_files()
    
    def setup_csv_files(self):
        """Configura los archivos CSV para guardar datos de aprendizaje"""
        # Archivo para estadísticas de partidas
        self.game_stats_file = 'neural_data/game_statistics.csv'
        if not os.path.exists(self.game_stats_file):
            with open(self.game_stats_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'game_number', 'result', 'moves_count', 'duration', 
                               'avg_evaluation', 'learning_rate', 'win_rate'])
        
        # Archivo para análisis de movimientos
        self.move_analysis_file = 'neural_data/move_analysis.csv'
        if not os.path.exists(self.move_analysis_file):
            with open(self.move_analysis_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'game_number', 'move_number', 'from_square', 'to_square', 
                               'piece_moved', 'piece_captured', 'neural_evaluation', 'traditional_evaluation',
                               'combined_evaluation', 'was_best_move'])
        
        # Archivo para evaluaciones de posiciones
        self.position_eval_file = 'neural_data/position_evaluations.csv'
        if not os.path.exists(self.position_eval_file):
            with open(self.position_eval_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'fen', 'neural_evaluation', 'traditional_evaluation', 
                               'game_phase', 'material_balance', 'piece_activity'])
        
        # Archivo para progreso del aprendizaje
        self.learning_progress_file = 'neural_data/learning_progress.csv'
        if not os.path.exists(self.learning_progress_file):
            with open(self.learning_progress_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'epoch', 'loss', 'accuracy', 'learning_rate', 
                               'total_games', 'win_rate'])
    
    def relu(self, x):
        """Función de activación ReLU"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivada de ReLU"""
        return (x > 0).astype(float)
    
    def leaky_relu(self, x, alpha=0.01):
        """Función de activación Leaky ReLU"""
        return np.maximum(alpha * x, x)
    
    def leaky_relu_derivative(self, x, alpha=0.01):
        """Derivada de Leaky ReLU"""
        return np.where(x > 0, 1, alpha)
    
    def sigmoid(self, x):
        """Función de activación sigmoide mejorada"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        """Derivada de la función sigmoide"""
        return x * (1 - x)
    
    def tanh(self, x):
        """Función de activación tanh"""
        return np.tanh(np.clip(x, -500, 500))
    
    def tanh_derivative(self, x):
        """Derivada de la función tanh"""
        return 1 - x**2
    
    def board_to_vector(self, board):
        """
        Convierte el tablero de ajedrez a un vector numérico con características avanzadas
        """
        vector = np.zeros(768)  # 8x8x12 para las piezas
        
        piece_to_index = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,  # Blancas
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11  # Negras
        }
        
        # Representación básica del tablero
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                row = square // 8
                col = square % 8
                piece_index = piece_to_index[piece.symbol()]
                vector_index = (row * 8 + col) * 12 + piece_index
                vector[vector_index] = 1
        
        return vector.reshape(1, -1)
    
    def extract_advanced_features(self, board):
        """Extrae características avanzadas del tablero"""
        features = {}
        
        # Características básicas con validación
        features['material_balance'] = self.calculate_material_balance(board) or 0
        features['piece_activity'] = self.calculate_piece_activity(board) or 0
        features['king_safety'] = self.calculate_king_safety(board) or 0
        features['pawn_structure'] = self.calculate_pawn_structure(board) or 0
        features['center_control'] = self.calculate_center_control(board) or 0
        features['game_phase'] = self.determine_game_phase(board) or 0
        
        # Características dinámicas
        features['in_check'] = board.is_check()
        features['checkmate_threat'] = self.has_checkmate_threat(board)
        features['mobility'] = len(list(board.legal_moves))
        features['castling_rights'] = self.get_castling_rights(board) or 0
        
        return features
    
    def calculate_material_balance(self, board):
        """Calcula el balance material"""
        piece_values = {'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0,
                       'p': -1, 'n': -3, 'b': -3, 'r': -5, 'q': -9, 'k': 0}
        
        balance = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                balance += piece_values[piece.symbol()]
        
        return balance
    
    def calculate_piece_activity(self, board):
        """Calcula la actividad de las piezas"""
        activity = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                # Piezas en el centro valen más
                center_bonus = 0
                file, rank = chess.square_file(square), chess.square_rank(square)
                if 2 <= file <= 5 and 2 <= rank <= 5:
                    center_bonus = 0.5
                if 3 <= file <= 4 and 3 <= rank <= 4:
                    center_bonus = 1.0
                
                activity += center_bonus * (1 if piece.color == chess.WHITE else -1)
        
        return activity
    
    def calculate_king_safety(self, board):
        """Evalúa la seguridad del rey"""
        white_king = board.king(chess.WHITE)
        black_king = board.king(chess.BLACK)
        
        safety = 0
        if white_king and black_king:
            # Penalizar reyes expuestos
            white_safety = self.king_safety_score(board, white_king, chess.WHITE)
            black_safety = self.king_safety_score(board, black_king, chess.BLACK)
            safety = white_safety - black_safety
        
        return safety
    
    def king_safety_score(self, board, king_square, color):
        """Calcula la puntuación de seguridad para un rey específico"""
        score = 0
        king_file, king_rank = chess.square_file(king_square), chess.square_rank(king_square)
        
        # Peones escudo
        pawn_shield = 0
        for file_offset in [-1, 0, 1]:
            shield_file = king_file + file_offset
            if 0 <= shield_file <= 7:
                for rank_offset in [1, 2] if color == chess.WHITE else [-1, -2]:
                    shield_rank = king_rank + rank_offset
                    if 0 <= shield_rank <= 7:
                        shield_square = chess.square(shield_file, shield_rank)
                        piece = board.piece_at(shield_square)
                        if piece and piece.piece_type == chess.PAWN and piece.color == color:
                            pawn_shield += 1
        
        score += pawn_shield * 10
        return score
    
    def calculate_pawn_structure(self, board):
        """Evalúa la estructura de peones"""
        structure = 0
        
        # Peones doblados, aislados, pasados
        white_pawns = []
        black_pawns = []
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN:
                file = chess.square_file(square)
                rank = chess.square_rank(square)
                if piece.color == chess.WHITE:
                    white_pawns.append((file, rank))
                else:
                    black_pawns.append((file, rank))
        
        # Evaluar peones doblados con validación
        doubled_pawns_score = self.evaluate_doubled_pawns(white_pawns, black_pawns)
        structure += doubled_pawns_score if doubled_pawns_score is not None else 0
        
        return structure
    
    def evaluate_doubled_pawns(self, white_pawns, black_pawns):
        """Evalúa peones doblados"""
        score = 0
        
        # Contar peones por columna
        white_files = {}
        black_files = {}
        
        for file, rank in white_pawns:
            white_files[file] = white_files.get(file, 0) + 1
        
        for file, rank in black_pawns:
            black_files[file] = black_files.get(file, 0) + 1
        
        # Penalizar peones doblados
        for file, count in white_files.items():
            if count > 1:
                score -= (count - 1) * 5
        
        for file, count in black_files.items():
            if count > 1:
                score += (count - 1) * 5
        
        return score
    
    def calculate_center_control(self, board):
        """Evalúa el control del centro"""
        center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
        control = 0
        
        for square in center_squares:
            piece = board.piece_at(square)
            if piece:
                control += 1 if piece.color == chess.WHITE else -1
        
        return control
    
    def determine_game_phase(self, board):
        """Determina la fase del juego (apertura, medio juego, final)"""
        piece_count = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type != chess.PAWN:
                piece_count += 1
        
        if piece_count > 20:
            return 0  # Apertura
        elif piece_count > 10:
            return 1  # Medio juego
        else:
            return 2  # Final
    
    def has_checkmate_threat(self, board):
        """Verifica si hay amenaza de jaque mate"""
        for move in board.legal_moves:
            temp_board = board.copy()
            temp_board.push(move)
            if temp_board.is_checkmate():
                return True
        return False
    
    def get_castling_rights(self, board):
        """Obtiene los derechos de enroque"""
        rights = 0
        if board.has_kingside_castling_rights(chess.WHITE):
            rights += 1
        if board.has_queenside_castling_rights(chess.WHITE):
            rights += 1
        if board.has_kingside_castling_rights(chess.BLACK):
            rights -= 1
        if board.has_queenside_castling_rights(chess.BLACK):
            rights -= 1
        return rights
    
    def forward(self, X):
        """Propagación hacia adelante con múltiples capas"""
        self.activations = [X]
        self.z_values = []
        
        current_input = X
        
        for i in range(len(self.weights)):
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            
            # Aplicar función de activación
            if i < len(self.weights) - 1:  # Capas ocultas
                activation = self.leaky_relu(z)
            else:  # Capa de salida
                activation = self.tanh(z)
            
            self.activations.append(activation)
            current_input = activation
        
        return self.activations[-1]
    
    def backward(self, X, y, output):
        """Propagación hacia atrás con múltiples capas"""
        m = X.shape[0]
        
        # Calcular gradientes
        gradients_w = []
        gradients_b = []
        
        # Error de la capa de salida
        delta = output - y
        
        # Retropropagación
        for i in range(len(self.weights) - 1, -1, -1):
            # Gradientes de pesos y bias
            dW = (1/m) * np.dot(self.activations[i].T, delta)
            db = (1/m) * np.sum(delta, axis=0, keepdims=True)
            
            gradients_w.insert(0, dW)
            gradients_b.insert(0, db)
            
            # Calcular delta para la capa anterior
            if i > 0:
                if i == len(self.weights) - 1:  # Capa de salida
                    delta = np.dot(delta, self.weights[i].T) * self.tanh_derivative(self.activations[i])
                else:  # Capas ocultas
                    delta = np.dot(delta, self.weights[i].T) * self.leaky_relu_derivative(self.activations[i])
        
        # Actualizar pesos y bias
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * gradients_w[i]
            self.biases[i] -= self.learning_rate * gradients_b[i]
    
    def evaluate_position(self, board):
        """Evalúa una posición del tablero con análisis avanzado"""
        board_vector = self.board_to_vector(board)
        neural_eval = self.forward(board_vector)
        
        # Extraer características avanzadas
        features = self.extract_advanced_features(board)
        
        # Combinar evaluación neural con características
        combined_eval = float(neural_eval[0, 0])
        
        # Ajustar basándose en características especiales
        if features['in_check']:
            combined_eval *= 1.2 if combined_eval < 0 else 0.8
        
        if features['checkmate_threat']:
            combined_eval *= 1.5 if combined_eval > 0 else 0.5
        
        # Guardar evaluación para análisis
        self.save_position_evaluation(board, combined_eval, features)
        
        return combined_eval
    
    def save_position_evaluation(self, board, evaluation, features):
        """Guarda la evaluación de posición en CSV"""
        try:
            with open(self.position_eval_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    board.fen(),
                    evaluation,
                    features.get('material_balance', 0),
                    features.get('game_phase', 0),
                    features.get('material_balance', 0),
                    features.get('piece_activity', 0)
                ])
        except Exception as e:
            print(f"Error guardando evaluación: {e}")
    
    def add_game_data(self, board, move, neural_eval, traditional_eval=0):
        """Añade datos de la partida con análisis detallado"""
        board_vector = self.board_to_vector(board)
        
        # Analizar el movimiento
        move_analysis = self.analyze_move(board, move, neural_eval, traditional_eval)
        
        self.training_data.append({
            'board': board_vector,
            'move': move,
            'neural_eval': neural_eval,
            'traditional_eval': traditional_eval,
            'combined_eval': 0.7 * neural_eval + 0.3 * traditional_eval,
            'features': self.extract_advanced_features(board),
            'analysis': move_analysis
        })
        
        # Guardar análisis del movimiento
        self.save_move_analysis(move_analysis)
    
    def analyze_move(self, board, move, neural_eval, traditional_eval):
        """Analiza un movimiento específico"""
        try:
            chess_move = chess.Move.from_uci(move)
            piece_moved = board.piece_at(chess_move.from_square)
            piece_captured = board.piece_at(chess_move.to_square)
            
            analysis = {
                'timestamp': datetime.now().isoformat(),
                'game_number': self.games_played + 1,
                'move_number': len(self.training_data) + 1,
                'from_square': chess.square_name(chess_move.from_square),
                'to_square': chess.square_name(chess_move.to_square),
                'piece_moved': piece_moved.symbol() if piece_moved else '',
                'piece_captured': piece_captured.symbol() if piece_captured else '',
                'neural_evaluation': neural_eval,
                'traditional_evaluation': traditional_eval,
                'combined_evaluation': 0.7 * neural_eval + 0.3 * traditional_eval,
                'was_best_move': False  # Se actualizará después
            }
            
            return analysis
        except Exception as e:
            print(f"Error analizando movimiento: {e}")
            return {}
    
    def save_move_analysis(self, analysis):
        """Guarda el análisis del movimiento en CSV"""
        if not analysis:
            return
        
        try:
            with open(self.move_analysis_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    analysis.get('timestamp', ''),
                    analysis.get('game_number', 0),
                    analysis.get('move_number', 0),
                    analysis.get('from_square', ''),
                    analysis.get('to_square', ''),
                    analysis.get('piece_moved', ''),
                    analysis.get('piece_captured', ''),
                    analysis.get('neural_evaluation', 0),
                    analysis.get('traditional_evaluation', 0),
                    analysis.get('combined_evaluation', 0),
                    analysis.get('was_best_move', False)
                ])
        except Exception as e:
            print(f"Error guardando análisis de movimiento: {e}")
    
    def train_from_game(self, game_result):
        """
        Entrena la red neuronal con técnicas avanzadas
        """
        if not self.training_data:
            return
        
        # Preparar datos de entrenamiento
        X = np.vstack([data['board'] for data in self.training_data])
        
        # Crear etiquetas más sofisticadas
        y = self.create_sophisticated_labels(game_result)
        
        # Entrenar con múltiples épocas
        losses = []
        for epoch in range(15):  # Más épocas para mejor aprendizaje
            output = self.forward(X)
            loss = np.mean((output - y) ** 2)
            losses.append(loss)
            
            self.backward(X, y, output)
            
            # Guardar progreso del aprendizaje
            if epoch % 5 == 0:
                self.save_learning_progress(epoch, loss)
        
        # Experience replay - entrenar con datos históricos
        self.experience_replay_training()
        
        # Añadir datos actuales al replay buffer
        self.add_to_experience_replay(X, y)
        
        # Actualizar estadísticas
        self.update_statistics(game_result)
        
        # Adaptar tasa de aprendizaje
        self.adapt_learning_rate()
        
        # Limpiar datos de entrenamiento
        self.training_data = []
        
        # Guardar modelo periodicamente
        if self.games_played % 3 == 0:
            self.save_model()
    
    def create_sophisticated_labels(self, game_result):
        """Crea etiquetas más sofisticadas para el entrenamiento"""
        labels = []
        
        # Descuento temporal más sofisticado
        for i, data in enumerate(self.training_data):
            # Factor de descuento basado en la posición temporal
            time_factor = 0.98 ** (len(self.training_data) - 1 - i)
            
            # Ajustar basándose en características del juego
            features = data.get('features', {})
            feature_adjustment = 1.0
            
            # Bonificar movimientos en posiciones críticas
            if features.get('in_check', False):
                feature_adjustment *= 1.3
            if features.get('checkmate_threat', False):
                feature_adjustment *= 1.5
            
            # Ajustar por fase del juego
            game_phase = features.get('game_phase', 1)
            if game_phase == 2:  # Final del juego
                feature_adjustment *= 1.2
            
            label = game_result * time_factor * feature_adjustment
            labels.append([label])
        
        return np.array(labels)
    
    def experience_replay_training(self):
        """Entrena con experiencias pasadas"""
        if len(self.experience_replay) < 32:  # Mínimo para entrenamiento
            return
        
        # Seleccionar muestra aleatoria
        batch_size = min(32, len(self.experience_replay))
        batch_indices = random.sample(range(len(self.experience_replay)), batch_size)
        
        batch_X = []
        batch_y = []
        
        for idx in batch_indices:
            experience = self.experience_replay[idx]
            batch_X.append(experience['X'])
            batch_y.append(experience['y'])
        
        if batch_X:
            X_batch = np.vstack(batch_X)
            y_batch = np.vstack(batch_y)
            
            # Entrenar con la muestra
            for _ in range(5):
                output = self.forward(X_batch)
                self.backward(X_batch, y_batch, output)
    
    def add_to_experience_replay(self, X, y):
        """Añade experiencias al buffer de replay"""
        for i in range(len(X)):
            experience = {
                'X': X[i].reshape(1, -1),
                'y': y[i].reshape(1, -1),
                'timestamp': datetime.now()
            }
            
            self.experience_replay.append(experience)
            
            # Mantener tamaño del buffer
            if len(self.experience_replay) > self.max_replay_size:
                self.experience_replay.pop(0)
    
    def adapt_learning_rate(self):
        """Adapta la tasa de aprendizaje basándose en el progreso"""
        if self.games_played > 0:
            # Reducir tasa de aprendizaje gradualmente
            self.learning_rate = max(
                self.min_learning_rate,
                self.initial_learning_rate * (self.learning_decay ** self.games_played)
            )
    
    def update_statistics(self, game_result):
        """Actualiza estadísticas del modelo"""
        self.games_played += 1
        self.total_moves += len(self.training_data)
        
        if game_result > 0:
            self.wins += 1
        elif game_result < 0:
            self.losses += 1
        else:
            self.draws += 1
        
        # Calcular tasa de victoria
        win_rate = (self.wins / self.games_played) * 100 if self.games_played > 0 else 0
        
        # Guardar estadísticas del juego
        self.save_game_statistics(game_result, win_rate)
    
    def save_game_statistics(self, result, win_rate):
        """Guarda estadísticas de la partida en CSV"""
        try:
            with open(self.game_stats_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    self.games_played,
                    result,
                    len(self.training_data),
                    0,  # duración (se puede implementar)
                    0,  # evaluación promedio
                    self.learning_rate,
                    win_rate
                ])
        except Exception as e:
            print(f"Error guardando estadísticas: {e}")
    
    def save_learning_progress(self, epoch, loss):
        """Guarda el progreso del aprendizaje"""
        try:
            accuracy = self.calculate_accuracy()
            win_rate = (self.wins / self.games_played) * 100 if self.games_played > 0 else 0
            
            with open(self.learning_progress_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    epoch,
                    loss,
                    accuracy,
                    self.learning_rate,
                    self.games_played,
                    win_rate
                ])
        except Exception as e:
            print(f"Error guardando progreso: {e}")
    
    def calculate_accuracy(self):
        """Calcula la precisión del modelo"""
        if self.total_moves > 0:
            return (self.correct_predictions / self.total_moves) * 100
        return 0.0
    
    def save_model(self):
        """Guarda el modelo avanzado"""
        model_data = {
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() for b in self.biases],
            'hidden_layers': self.hidden_layers,
            'games_played': self.games_played,
            'wins': self.wins,
            'losses': self.losses,
            'draws': self.draws,
            'total_moves': self.total_moves,
            'correct_predictions': self.correct_predictions,
            'learning_rate': self.learning_rate,
            'initial_learning_rate': self.initial_learning_rate,
            'experience_replay_size': len(self.experience_replay)
        }
        
        # Guardar modelo principal
        with open('neural_data/advanced_chess_model.json', 'w') as f:
            json.dump(model_data, f, indent=2)
        
        # Guardar experience replay por separado
        replay_data = []
        for exp in self.experience_replay[-1000:]:  # Guardar solo los últimos 1000
            replay_data.append({
                'X': exp['X'].tolist(),
                'y': exp['y'].tolist(),
                'timestamp': exp['timestamp'].isoformat()
            })
        
        with open('neural_data/experience_replay.json', 'w') as f:
            json.dump(replay_data, f, indent=2)
        
        print(f"✅ Modelo avanzado guardado - Partidas: {self.games_played}, Victorias: {self.wins}, Derrotas: {self.losses}")
        win_rate = (self.wins/self.games_played)*100 if self.games_played > 0 else 0
        print(f"📊 Tasa de victoria: {win_rate:.1f}%, Experiencias: {len(self.experience_replay)}")
    
    def load_model(self):
        """Carga el modelo avanzado"""
        try:
            # Cargar modelo principal
            with open('neural_data/advanced_chess_model.json', 'r') as f:
                model_data = json.load(f)
            
            # Restaurar pesos y bias
            self.weights = [np.array(w) for w in model_data['weights']]
            self.biases = [np.array(b) for b in model_data['biases']]
            
            # Restaurar estadísticas
            self.games_played = model_data.get('games_played', 0)
            self.wins = model_data.get('wins', 0)
            self.losses = model_data.get('losses', 0)
            self.draws = model_data.get('draws', 0)
            self.total_moves = model_data.get('total_moves', 0)
            self.correct_predictions = model_data.get('correct_predictions', 0)
            self.learning_rate = model_data.get('learning_rate', self.initial_learning_rate)
            
            # Cargar experience replay
            try:
                with open('neural_data/experience_replay.json', 'r') as f:
                    replay_data = json.load(f)
                
                self.experience_replay = []
                for exp_data in replay_data:
                    experience = {
                        'X': np.array(exp_data['X']),
                        'y': np.array(exp_data['y']),
                        'timestamp': datetime.fromisoformat(exp_data['timestamp'])
                    }
                    self.experience_replay.append(experience)
                
                print(f"📚 Experience replay cargado: {len(self.experience_replay)} experiencias")
            except:
                print("⚠️ No se pudo cargar experience replay, iniciando vacío")
            
            win_rate = (self.wins / self.games_played) * 100 if self.games_played > 0 else 0
            print(f"✅ Modelo avanzado cargado:")
            print(f"   🎮 Partidas: {self.games_played}")
            print(f"   🏆 Victorias: {self.wins} | Derrotas: {self.losses} | Empates: {self.draws}")
            print(f"   📈 Tasa de victoria: {win_rate:.1f}%")
            print(f"   🧠 Tasa de aprendizaje: {self.learning_rate:.6f}")
            
        except FileNotFoundError:
            print("🆕 No se encontró modelo previo. Iniciando red neuronal avanzada nueva.")
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
    
    def get_stats(self):
        """Devuelve estadísticas detalladas del modelo"""
        if self.games_played > 0:
            win_rate = (self.wins / self.games_played) * 100
            accuracy = self.calculate_accuracy()
            
            return {
                'games_played': self.games_played,
                'wins': self.wins,
                'losses': self.losses,
                'draws': self.draws,
                'win_rate': win_rate,
                'total_moves': self.total_moves,
                'accuracy': accuracy,
                'learning_rate': self.learning_rate,
                'experience_replay_size': len(self.experience_replay)
            }
        return {
            'games_played': 0, 'wins': 0, 'losses': 0, 'draws': 0, 
            'win_rate': 0, 'total_moves': 0, 'accuracy': 0, 
            'learning_rate': self.learning_rate, 'experience_replay_size': 0
        }
    
    def analyze_learning_progress(self):
        """Analiza el progreso del aprendizaje desde los archivos CSV"""
        try:
            # Leer estadísticas de juegos
            if os.path.exists(self.game_stats_file):
                df_games = pd.read_csv(self.game_stats_file)
                print("\n📊 ANÁLISIS DE PROGRESO:")
                print(f"Total de partidas registradas: {len(df_games)}")
                
                if len(df_games) > 0:
                    recent_games = df_games.tail(10)
                    recent_win_rate = recent_games['win_rate'].iloc[-1] if not recent_games.empty else 0
                    print(f"Tasa de victoria reciente: {recent_win_rate:.1f}%")
                    
                    # Mostrar evolución
                    if len(df_games) >= 5:
                        early_win_rate = df_games.head(5)['win_rate'].mean()
                        improvement = recent_win_rate - early_win_rate
                        print(f"Mejora desde el inicio: {improvement:+.1f}%")
            
            # Leer progreso de aprendizaje
            if os.path.exists(self.learning_progress_file):
                df_progress = pd.read_csv(self.learning_progress_file)
                if len(df_progress) > 0:
                    latest_loss = df_progress['loss'].iloc[-1]
                    latest_accuracy = df_progress['accuracy'].iloc[-1]
                    print(f"Pérdida actual: {latest_loss:.4f}")
                    print(f"Precisión actual: {latest_accuracy:.1f}%")
            
        except Exception as e:
            print(f"Error analizando progreso: {e}")
    
    def export_training_summary(self):
        """Exporta un resumen del entrenamiento"""
        summary = {
            'model_info': {
                'architecture': f"Input({self.input_size}) -> Hidden{self.hidden_layers} -> Output({self.output_size})",
                'total_parameters': sum(w.size for w in self.weights) + sum(b.size for b in self.biases),
                'activation_functions': 'LeakyReLU (hidden), Tanh (output)'
            },
            'training_stats': self.get_stats(),
            'csv_files': {
                'game_statistics': self.game_stats_file,
                'move_analysis': self.move_analysis_file,
                'position_evaluations': self.position_eval_file,
                'learning_progress': self.learning_progress_file
            }
        }
        
        with open('neural_data/training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary

# Función auxiliar mejorada
def neural_evaluate_move(board, move, neural_net):
    """Evalúa un movimiento usando la red neuronal avanzada"""
    temp_board = board.copy()
    temp_board.push(move)
    return neural_net.evaluate_position(temp_board)
