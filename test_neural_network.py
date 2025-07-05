#!/usr/bin/env python3
"""
Script de prueba para verificar las funciones de la red neuronal
"""
import chess
from chess_neural_network import AdvancedChessNeuralNetwork

# Crear instancia de la red neuronal
neural_net = AdvancedChessNeuralNetwork()

# Crear tablero de prueba
board = chess.Board()

print("🧪 === PRUEBA DE FUNCIONES DE LA RED NEURONAL ===")
print(f"Tablero inicial: {board.fen()}")

# Probar todas las funciones individualmente
try:
    print("\n🔍 Probando funciones individuales:")
    
    # Probar calculate_material_balance
    material_balance = neural_net.calculate_material_balance(board)
    print(f"✅ Material balance: {material_balance}")
    
    # Probar calculate_piece_activity
    piece_activity = neural_net.calculate_piece_activity(board)
    print(f"✅ Piece activity: {piece_activity}")
    
    # Probar calculate_king_safety
    king_safety = neural_net.calculate_king_safety(board)
    print(f"✅ King safety: {king_safety}")
    
    # Probar calculate_pawn_structure
    pawn_structure = neural_net.calculate_pawn_structure(board)
    print(f"✅ Pawn structure: {pawn_structure}")
    
    # Probar calculate_center_control
    center_control = neural_net.calculate_center_control(board)
    print(f"✅ Center control: {center_control}")
    
    # Probar determine_game_phase
    game_phase = neural_net.determine_game_phase(board)
    print(f"✅ Game phase: {game_phase}")
    
    # Probar get_castling_rights
    castling_rights = neural_net.get_castling_rights(board)
    print(f"✅ Castling rights: {castling_rights}")
    
    # Probar extract_advanced_features
    features = neural_net.extract_advanced_features(board)
    print(f"✅ Advanced features: {features}")
    
    # Probar evaluate_position
    position_eval = neural_net.evaluate_position(board)
    print(f"✅ Position evaluation: {position_eval}")
    
    print("\n✅ Todas las funciones funcionan correctamente!")
    
except Exception as e:
    print(f"❌ Error en la prueba: {e}")
    import traceback
    traceback.print_exc()

print("\n🎯 Probando evaluación de movimientos:")
try:
    # Probar evaluación de un movimiento específico
    move = chess.Move.from_uci("e2e4")
    temp_board = board.copy()
    temp_board.push(move)
    move_eval = neural_net.evaluate_position(temp_board)
    print(f"✅ Evaluación del movimiento e2e4: {move_eval}")
    
    # Probar todos los movimientos legales
    print("\n🔄 Probando evaluación de todos los movimientos legales:")
    legal_moves = list(board.legal_moves)
    for i, move in enumerate(legal_moves[:5]):  # Solo los primeros 5 para no saturar
        temp_board = board.copy()
        temp_board.push(move)
        move_eval = neural_net.evaluate_position(temp_board)
        print(f"  {move}: {move_eval}")
        
    print("✅ Evaluación de movimientos funciona correctamente!")
    
except Exception as e:
    print(f"❌ Error evaluando movimientos: {e}")
    import traceback
    traceback.print_exc()

print("\n🎮 Prueba completada.")
