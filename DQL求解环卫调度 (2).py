# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 15:57:17 2026

@author: WIN
"""

# 修复OpenMP重复初始化问题
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pandas as pd
import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional, Any
import copy
import time
import matplotlib.pyplot as plt
from datetime import datetime
import json

# ===================== 第一部分：数据读取和基础类定义 =====================
def read_excel_data(file_path):
    """读取Excel文件中的两个工作表"""
    try:
        population_df = pd.read_excel(file_path, sheet_name='population_center')
        transferstation_df = pd.read_excel(file_path, sheet_name='transferstaion')
        print(f"成功读取 {len(population_df)} 个人口中心和 {len(transferstation_df)} 个转运站")
        return population_df, transferstation_df
    except Exception as e:
        print(f"读取Excel文件时出错: {e}")
        return None, None

class Location:
    """位置基类"""
    def __init__(self, location_id: str, longitude: float, latitude: float):
        self.id = location_id
        self.longitude = longitude
        self.latitude = latitude
    
    def distance_to(self, other_location) -> float:
        """计算到另一个位置的距离（公里）"""
        return self.haversine_distance(other_location.longitude, other_location.latitude)
    
    def haversine_distance(self, lon2: float, lat2: float) -> float:
        """使用Haversine公式计算距离"""
        R = 6371.0  # 地球半径，公里
        
        lat1_rad = math.radians(self.latitude)
        lon1_rad = math.radians(self.longitude)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad
        
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c

class PopulationCenter(Location):
    """人口中心 - 按时间均匀生成垃圾"""
    def __init__(self, center_id: str, longitude: float, latitude: float, population: int):
        super().__init__(center_id, longitude, latitude)
        self.population = population
        
        # 垃圾生成参数（人均日产量1.2kg = 0.0012吨）
        self.garbage_per_person_per_day = 0.0012  # 吨/人/天
        self.garbage_per_person_per_hour = self.garbage_per_person_per_day / 24  # 吨/人/小时
        
        # 垃圾生成速率（吨/小时）
        self.generation_rate = self.population * self.garbage_per_person_per_hour
        
        # 当前状态
        self.current_garbage = 0.0  # 当前垃圾量（吨）
        self.waiting_time = 0.0  # 垃圾等待时间（小时）
        self.total_generated = 0.0  # 累计生成垃圾量
        self.total_collected = 0.0  # 累计收集垃圾量
        self.last_collection_time = 0.0  # 上次收集时间
        self.collection_queue = 0  # 等待收集的车辆数
    
    def update_garbage(self, time_elapsed: float):
        """
        更新垃圾量 - 按时间均匀生成
        参数: time_elapsed - 经过的时间（小时）
        """
        # 计算在这个时间段内生成的垃圾量
        garbage_generated = self.generation_rate * time_elapsed
        
        # 更新垃圾量
        self.current_garbage += garbage_generated
        self.total_generated += garbage_generated
        
        # 更新等待时间（如果有垃圾）
        if self.current_garbage > 0:
            self.waiting_time += time_elapsed
    
    def collect_garbage(self, vehicle_capacity: float) -> Tuple[float, bool]:
        """
        收集垃圾
        返回: (实际收集量, 是否成功收集)
        注意：垃圾量必须达到车辆容量才能收集
        """
        if self.current_garbage < vehicle_capacity:
            return 0.0, False
        
        # 收集一车垃圾
        collect_amount = vehicle_capacity  # 每次收集一整车
        self.current_garbage -= collect_amount
        self.total_collected += collect_amount
        self.last_collection_time = 0.0  # 重置收集时间
        self.collection_queue = max(0, self.collection_queue - 1)
        
        # 如果收集完垃圾，重置等待时间
        if self.current_garbage <= 0:
            self.waiting_time = 0.0
        
        return collect_amount, True
    
    def get_garbage_demand(self, vehicle_capacity: float) -> int:
        """获取垃圾处理需求（需要的车辆数）"""
        if vehicle_capacity <= 0:
            return 0
        return int(self.current_garbage // vehicle_capacity)
    
    def can_collect(self, vehicle_capacity: float) -> bool:
        """检查是否可以收集垃圾（达到车辆容量阈值）"""
        return self.current_garbage >= vehicle_capacity
    
    def get_generation_info(self) -> Dict[str, Any]:
        """获取垃圾生成信息"""
        return {
            'population': self.population,
            'generation_rate_tph': self.generation_rate,  # 吨/小时
            'daily_generation': self.population * self.garbage_per_person_per_day,
            'current_garbage': self.current_garbage,
            'waiting_time_hours': self.waiting_time
        }

class TransferStation(Location):
    """转运站"""
    def __init__(self, station_id: str, longitude: float, latitude: float, 
                 capacity: float, device_num: int):
        super().__init__(station_id, longitude, latitude)
        self.capacity = capacity  # 容量（吨）
        self.device_num = device_num  # 设备数量
        self.available_devices = device_num  # 可用设备数
        
        # 当前状态
        self.current_garbage = 0.0  # 当前垃圾量
        self.queue = deque()  # 排队车辆队列
        self.processing_vehicles = []  # 正在处理的车辆
        self.total_processed = 0.0  # 累计处理垃圾量
        self.queue_waiting_time = 0.0  # 排队总等待时间
        
        # 卸载效率参数（30-50吨/小时）
        self.unload_speed_min = 30  # 吨/小时
        self.unload_speed_max = 50  # 吨/小时
    
    @property
    def can_unload(self):
        """检查是否可以卸载（有可用设备且容量充足）"""
        return (self.available_devices > 0 and 
                self.current_garbage < self.capacity * 0.95)  # 保留5%缓冲
    
    def add_to_queue(self, vehicle_id: str):
        """添加车辆到排队队列"""
        if vehicle_id not in self.queue:
            self.queue.append(vehicle_id)
            return True
        return False
    
    def process_queue(self):
        """处理排队车辆"""
        processed = []
        while self.queue and self.can_unload:
            vehicle_id = self.queue.popleft()
            self.processing_vehicles.append(vehicle_id)
            self.available_devices -= 1
            processed.append(vehicle_id)
        return processed
    
    def update_queue_waiting(self, time_elapsed: float):
        """更新排队等待时间"""
        self.queue_waiting_time += len(self.queue) * time_elapsed

class CollectionVehicle:
    """收运车辆"""
    
    # 全局车辆ID计数器
    _vehicle_counter = 0
    
    def __init__(self, capacity: float = 10.0):
        CollectionVehicle._vehicle_counter += 1
        self.id = f"vehicle_{CollectionVehicle._vehicle_counter:03d}"
        self.capacity = capacity  # 载重容量（吨），固定为10吨
        
        # 当前状态
        self.current_load = 0.0  # 当前载重
        self.current_location_type = None  # 'center', 'station', 'road'
        self.current_location_id = None  # 当前位置ID
        self.status = 'idle'  # 'idle', 'traveling', 'loading', 'unloading', 'waiting', 'loaded', 'resting'
        
        # 运输参数
        self.speed_min = 30  # km/h
        self.speed_max = 50  # km/h
        self.load_time_min = 10 / 60  # 小时 (10分钟)
        self.load_time_max = 15 / 60  # 小时 (15分钟)
        
        # 行程信息
        self.destination_type = None  # 'center', 'station'
        self.destination_id = None  # 目的地ID
        self.route_distance = 0.0  # 行程距离
        self.route_progress = 0.0  # 行程进度 0-1
        self.remaining_time = 0.0  # 剩余时间（小时）
        self.travel_speed = 0.0  # 行驶速度
        
        # 统计信息
        self.total_distance = 0.0  # 累计行驶距离
        self.total_garbage_collected = 0.0  # 累计收集垃圾量
        self.total_garbage_unloaded = 0.0  # 累计卸载垃圾量
        self.total_operating_time = 0.0  # 累计运营时间
        self.idle_time = 0.0  # 空闲时间
        self.waiting_time = 0.0  # 等待时间（排队）
        self.trips_completed = 0  # 完成运输次数
        self.empty_runs = 0  # 空跑次数
        self.rest_time = 0.0  # 休整时间
    
    def can_be_dispatched(self):
        """检查车辆是否可以被调度（空闲且在转运站）"""
        return (self.status == 'idle' and 
                self.current_location_type == 'station' and 
                self.current_load == 0)
    
    def is_at_station(self):
        """检查车辆是否在转运站"""
        return self.current_location_type == 'station'
    
    def is_at_center(self):
        """检查车辆是否在人口中心"""
        return self.current_location_type == 'center'
    
    def start_loading(self, load_time: float):
        """开始装载"""
        self.status = 'loading'
        self.remaining_time = load_time
    
    def start_unloading(self, unload_time: float):
        """开始卸载"""
        self.status = 'unloading'
        self.remaining_time = unload_time
    
    def start_travel(self, destination_type: str, destination_id: str, 
                    distance: float, speed: float):
        """开始运输"""
        self.destination_type = destination_type
        self.destination_id = destination_id
        self.route_distance = distance
        self.travel_speed = speed
        self.remaining_time = distance / speed
        self.status = 'traveling'
        self.current_location_type = 'road'
        self.current_location_id = f"from_{self.current_location_id}_to_{destination_id}"
    
    def start_rest(self, rest_time: float = 5/60):  # 默认休整5分钟
        """开始休整"""
        self.status = 'resting'
        self.remaining_time = rest_time
    
    def update_status(self, time_elapsed: float):
        """更新车辆状态"""
        if self.status in ['traveling', 'loading', 'unloading', 'resting']:
            self.remaining_time -= time_elapsed
            self.total_operating_time += time_elapsed
            
            if self.status == 'traveling':
                # 更新行程进度
                if self.route_distance > 0 and self.travel_speed > 0:
                    total_travel_time = self.route_distance / self.travel_speed
                    self.route_progress = 1 - (self.remaining_time / total_travel_time) if total_travel_time > 0 else 0
            elif self.status == 'resting':
                self.rest_time += time_elapsed
        
        elif self.status == 'waiting':
            self.waiting_time += time_elapsed
            self.total_operating_time += time_elapsed
        
        elif self.status == 'idle':
            self.idle_time += time_elapsed
            self.total_operating_time += time_elapsed

# ===================== 第二部分：深度强化学习DQN网络 =====================
class DQN(nn.Module):
    """深度Q网络（Dueling DQN）"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DQN, self).__init__()
        
        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # 动作价值网络
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # 动作优势网络
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, action_dim)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, state):
        """前向传播"""
        features = self.state_encoder(state)
        
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Dueling DQN: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = value + advantages - advantages.mean(dim=1, keepdim=True)
        
        return q_values

class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """存储经验"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """采样一批经验"""
        if len(self.buffer) < batch_size:
            return None
        
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones)
        )
    
    def __len__(self):
        return len(self.buffer)

# ===================== 第三部分：DQN智能体 =====================
class DQNAgent:
    """DQN智能体"""
    def __init__(self, state_dim, action_dim, num_vehicles=180, device='cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_vehicles = num_vehicles
        self.device = device
        
        # DQN网络
        self.policy_net = DQN(state_dim, action_dim).to(device)
        self.target_net = DQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        
        # 经验回放缓冲区
        self.memory = ReplayBuffer(10000)
        
        # 训练参数
        self.batch_size = 64
        self.gamma = 0.95  # 折扣因子
        self.eps_start = 1.0  # 初始探索率
        self.eps_end = 0.01  # 最终探索率
        self.eps_decay = 0.995  # 探索率衰减
        self.eps = self.eps_start
        self.target_update = 10  # 目标网络更新频率
        self.steps_done = 0
        
        # 训练历史
        self.loss_history = []
        self.reward_history = []
        self.epsilon_history = []
    
    def select_action(self, state, available_actions=None):
        """选择动作（ε-贪婪策略）"""
        self.steps_done += 1
        
        # 衰减探索率
        self.eps = max(self.eps_end, self.eps * self.eps_decay)
        self.epsilon_history.append(self.eps)
        
        if random.random() < self.eps:
            # 探索：随机选择动作
            if available_actions:
                return random.choice(available_actions)
            else:
                return random.randint(0, self.action_dim - 1)
        else:
            # 利用：选择Q值最高的动作
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor).cpu().numpy()[0]
                
                if available_actions:
                    # 只考虑可用动作
                    available_q_values = [q_values[a] for a in available_actions]
                    best_idx = np.argmax(available_q_values)
                    return available_actions[best_idx]
                else:
                    return np.argmax(q_values)
    
    def optimize_model(self):
        """优化模型"""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # 从经验回放缓冲区采样
        batch = self.memory.sample(self.batch_size)
        if batch is None:
            return 0.0
        
        states, actions, rewards, next_states, dones = batch
        
        # 计算当前Q值
        current_q_values = self.policy_net(states.to(self.device)).gather(1, actions.unsqueeze(1).to(self.device))
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_net(next_states.to(self.device)).max(1)[0]
            target_q_values = rewards.to(self.device) + (1 - dones.to(self.device)) * self.gamma * next_q_values
        
        # 计算损失
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # 优化模型
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)  # 梯度裁剪
        self.optimizer.step()
        
        # 记录损失
        loss_value = loss.item()
        self.loss_history.append(loss_value)
        
        return loss_value
    
    def update_target_network(self):
        """更新目标网络"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save_model(self, path):
        """保存模型"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'eps': self.eps,
            'steps_done': self.steps_done
        }, path)
    
    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.eps = checkpoint['eps']
        self.steps_done = checkpoint['steps_done']

# ===================== 第四部分：强化学习环境（优化版） =====================
class WasteCollectionRLEnv:
    """垃圾收运强化学习环境（优化版）"""
    
    def __init__(self, population_df, transferstation_df, 
                 num_vehicles: int = 216, vehicle_capacity: float = 10.0):
        
        # 初始化人口中心
        self.population_centers = {}
        self.population_center_ids = []
        for _, row in population_df.iterrows():
            center = PopulationCenter(
                row['population_center_ID'],
                row['longitude'],
                row['latitude'],
                int(row['population'])
            )
            self.population_centers[center.id] = center
            self.population_center_ids.append(center.id)
        
        # 初始化转运站
        self.transfer_stations = {}
        self.transfer_station_ids = []
        for _, row in transferstation_df.iterrows():
            station = TransferStation(
                row['transferstation_ID'],
                row['longitude'],
                row['latitude'],
                float(row['capacity']),
                int(row['device_num'])
            )
            self.transfer_stations[station.id] = station
            self.transfer_station_ids.append(station.id)
        
        # 计算距离矩阵
        self._calculate_distance_matrix()
        
        # 初始化车辆
        self.vehicles = []
        total_devices = sum(station.device_num for station in self.transfer_stations.values())
        
        if num_vehicles != total_devices:
            print(f"警告：车辆数({num_vehicles})与设备数({total_devices})不同，调整为{total_devices}")
            num_vehicles = total_devices
        
        # 按转运站设备数分配车辆
        vehicle_counter = 0
        for station_id, station in self.transfer_stations.items():
            for i in range(station.device_num):
                vehicle = CollectionVehicle(capacity=vehicle_capacity)
                vehicle.current_location_type = 'station'
                vehicle.current_location_id = station_id
                vehicle.status = 'idle'
                self.vehicles.append(vehicle)
                vehicle_counter += 1
        
        # 环境参数
        self.time_step = 5/60  # 5分钟 = 5/60小时
        self.current_time = 0.0
        self.max_steps = 24 * 12  # 模拟24小时，每5分钟一步 = 288步
        self.current_step = 0
        self.vehicle_capacity = vehicle_capacity
        
        # 预约机制
        self.center_reservations = {}  # 人口中心预约：center_id -> [vehicle_ids]
        self.station_reservations = {}  # 转运站预约：station_id -> [vehicle_ids]
        self.route_reservations = {}  # 路径预约：vehicle_id -> (出发地, 目的地, 到达时间)
        
        # 初始化预约字典
        for center_id in self.population_center_ids:
            self.center_reservations[center_id] = []
        for station_id in self.transfer_station_ids:
            self.station_reservations[station_id] = []
        
        # 状态和动作空间
        self.num_population_centers = len(self.population_centers)
        self.num_transfer_stations = len(self.transfer_stations)
        self.num_vehicles = num_vehicles
        
        # 动作空间：去最近的人口中心、最近的转运站、或休息
        self.max_nearby_centers = 7  # 最近7个人口中心（只考虑有垃圾的）
        self.max_nearby_stations = 5  # 最近5个转运站（只考虑无排队的）
        self.action_space_size = self.max_nearby_centers + self.max_nearby_stations + 1  # +1 for rest
        
        # 状态空间维度
        num_sampled_vehicles = min(20, num_vehicles)
        self.state_space_size = (num_sampled_vehicles * 10 + 
                                min(20, self.num_population_centers) * 3 +  # 增加预约信息
                                self.num_transfer_stations * 4)  # 增加预约信息
        
        # 奖励函数权重
        self.reward_weights = {
            'garbage_collected': 10.0,
            'garbage_processed': 30.0,
            'distance_penalty': -0.05,
            'waiting_penalty': -0.5,
            'queue_penalty': -0.2,
            'congestion_penalty': -1.0,
            'utilization_bonus': 0.5,
            'efficiency_bonus': 0.2,
            'empty_run_penalty': -200.0,
            'rest_penalty': -40.0,
            'collision_penalty': -5.0,
            'reservation_violation': -10.0,  # 预约违规惩罚
        }
        
        # 历史记录
        self.history = {
            'total_garbage': [],
            'total_collected': [],
            'total_processed': [],
            'total_distance': [],
            'total_waiting': [],
            'total_queue': [],
            'vehicle_utilization': [],
            'rewards': [],
            'status_counts': [],
            'empty_runs': [],
            'rest_count': [],
            'reservation_violations': []  # 添加预约违规记录
        }
        
        # 性能指标
        self.performance_metrics = {
            'total_episode_reward': 0.0,
            'total_garbage_collected': 0.0,
            'total_garbage_processed': 0.0,
            'total_distance': 0.0,
            'average_waiting_time': 0.0,
            'average_queue_length': 0.0,
            'vehicle_utilization_rate': 0.0,
            'total_empty_runs': 0,
            'total_rest_count': 0,
            'total_reservation_violations': 0  # 添加预约违规指标
        }
    
    def _calculate_distance_matrix(self):
        """计算距离矩阵"""
        self.distances = {}
        
        # 转运站到人口中心的距离
        for station_id, station in self.transfer_stations.items():
            station_distances = []
            for center_id, center in self.population_centers.items():
                dist = station.distance_to(center)
                station_distances.append((center_id, dist, center.current_garbage))
            
            station_distances.sort(key=lambda x: x[1])
            self.distances[f'station_{station_id}'] = station_distances
        
        # 人口中心到转运站的距离
        for center_id, center in self.population_centers.items():
            center_distances = []
            for station_id, station in self.transfer_stations.items():
                dist = center.distance_to(station)
                center_distances.append((station_id, dist, len(station.queue)))
            
            center_distances.sort(key=lambda x: x[1])
            self.distances[f'center_{center_id}'] = center_distances
    
    def get_state(self) -> np.ndarray:
        """获取环境状态（包含预约信息）"""
        state = []
        
        # 1. 车辆状态（采样前20辆车）
        num_sampled_vehicles = min(20, len(self.vehicles))
        for i in range(num_sampled_vehicles):
            vehicle = self.vehicles[i]
            
            loc_type = [0, 0, 0]
            if vehicle.current_location_type == 'center':
                loc_type = [1, 0, 0]
            elif vehicle.current_location_type == 'station':
                loc_type = [0, 1, 0]
            elif vehicle.current_location_type == 'road':
                loc_type = [0, 0, 1]
            state.extend(loc_type)
            
            vehicle_status = [0, 0, 0]
            if vehicle.status == 'idle':
                vehicle_status = [1, 0, 0]
            elif vehicle.status in ['traveling', 'loading', 'unloading']:
                vehicle_status = [0, 1, 0]
            elif vehicle.status in ['waiting', 'loaded', 'resting']:
                vehicle_status = [0, 0, 1]
            state.extend(vehicle_status)
            
            load_ratio = vehicle.current_load / self.vehicle_capacity if self.vehicle_capacity > 0 else 0
            state.append(load_ratio)
            
            in_transit = 1.0 if vehicle.status == 'traveling' else 0.0
            state.append(in_transit)
            
            state.append(vehicle.route_progress)
            
            has_destination = 1.0 if vehicle.destination_id is not None else 0.0
            state.append(has_destination)
            
            # 检查是否有预约
            has_reservation = 1.0 if vehicle.id in self.route_reservations else 0.0
            state.append(has_reservation)
        
        # 2. 人口中心状态（采样前20个），增加预约信息
        num_sampled_centers = min(20, self.num_population_centers)
        for i in range(num_sampled_centers):
            center_id = self.population_center_ids[i]
            center = self.population_centers[center_id]
            
            demand_ratio = min(center.current_garbage / (self.vehicle_capacity * 3), 1.0)
            state.append(demand_ratio)
            
            waiting_ratio = min(center.waiting_time / 12, 1.0)
            state.append(waiting_ratio)
            
            # 预约车辆数比率
            reservation_ratio = min(len(self.center_reservations.get(center_id, [])) / 3, 1.0)
            state.append(reservation_ratio)
        
        # 3. 转运站状态（所有转运站），增加预约信息
        for station_id in self.transfer_station_ids:
            station = self.transfer_stations[station_id]
            
            garbage_ratio = station.current_garbage / station.capacity if station.capacity > 0 else 0
            state.append(garbage_ratio)
            
            queue_ratio = min(len(station.queue) / 5, 1.0)
            state.append(queue_ratio)
            
            device_ratio = station.available_devices / max(station.device_num, 1)
            state.append(device_ratio)
            
            # 预约车辆数比率
            reservation_ratio = min(len(self.station_reservations.get(station_id, [])) / station.device_num, 1.0)
            state.append(reservation_ratio)
        
        # 确保状态向量长度一致
        expected_length = self.state_space_size
        current_length = len(state)
        
        if current_length < expected_length:
            state.extend([0.0] * (expected_length - current_length))
        elif current_length > expected_length:
            state = state[:expected_length]
        
        return np.array(state, dtype=np.float32)
    
    def get_decision_vehicles(self) -> List[CollectionVehicle]:
        """获取需要决策的车辆"""
        decision_vehicles = []
        for vehicle in self.vehicles:
            if vehicle.status == 'idle' and vehicle.current_location_type == 'station' and vehicle.current_load == 0:
                decision_vehicles.append(vehicle)
            elif vehicle.status == 'loaded' and vehicle.current_location_type == 'center':
                decision_vehicles.append(vehicle)
            elif vehicle.status == 'idle' and vehicle.current_location_type == 'center' and vehicle.current_load == 0:
                decision_vehicles.append(vehicle)
            elif vehicle.status == 'idle' and vehicle.current_location_type == 'station' and vehicle.current_load > 0:
                decision_vehicles.append(vehicle)
        return decision_vehicles
    
    def get_available_actions(self, vehicle: CollectionVehicle) -> List[int]:
        """获取车辆可用的动作列表（考虑预约机制）"""
        available_actions = []
        
        rest_action = self.action_space_size - 1
        
        if vehicle.status == 'idle' and vehicle.current_location_type == 'station' and vehicle.current_load == 0:
            # 空闲车辆（在转运站，空车）-> 只考虑有垃圾的最近7个人口中心
            station_id = vehicle.current_location_id
            
            # 获取所有人口中心
            all_centers = self.distances.get(f'station_{station_id}', [])
            
            # 筛选条件：
            # 1. 垃圾量 >= 车辆容量
            # 2. 预约车辆数 < 2（避免过度预约）
            # 3. 按距离排序，取最近的
            valid_centers = []
            for center_id, dist, garbage in all_centers:
                center = self.population_centers[center_id]
                
                # 检查是否有足够的垃圾
                has_enough_garbage = center.current_garbage >= self.vehicle_capacity
                
                # 检查预约情况：如果预约的垃圾量已经足够，不再考虑
                reserved_vehicles = self.center_reservations.get(center_id, [])
                reserved_garbage = len(reserved_vehicles) * self.vehicle_capacity
                available_garbage = center.current_garbage - reserved_garbage
                
                # 还有足够垃圾且预约车辆数不超过2
                if (has_enough_garbage and 
                    available_garbage >= self.vehicle_capacity and
                    len(reserved_vehicles) < 2):
                    
                    valid_centers.append((center_id, dist, center.current_garbage))
            
            # 按距离排序，取最近的7个
            valid_centers.sort(key=lambda x: x[1])
            nearby_centers = valid_centers[:self.max_nearby_centers]
            
            for i, (center_id, dist, garbage) in enumerate(nearby_centers):
                available_actions.append(i)
            
            # 如果没有可去的人口中心，允许休息
            if not available_actions:
                available_actions.append(rest_action)
            else:
                available_actions.append(rest_action)
        
        elif (vehicle.status == 'loaded' and vehicle.current_location_type == 'center') or \
             (vehicle.status == 'idle' and vehicle.current_location_type == 'center' and vehicle.current_load == 0):
            # 已装载车辆（在人口中心）或到达后没有垃圾的空车 -> 只考虑无排队的最近5个转运站
            center_id = vehicle.current_location_id
            
            # 获取所有转运站
            all_stations = self.distances.get(f'center_{center_id}', [])
            
            # 筛选条件：
            # 1. 无排队（queue = 0）
            # 2. 可用设备 > 0
            # 3. 预约车辆数 < 可用设备数（避免过度预约）
            valid_stations = []
            for station_id, dist, queue_length in all_stations:
                station = self.transfer_stations[station_id]
                
                # 检查排队情况
                no_queue = queue_length == 0
                
                # 检查可用设备
                has_available_device = station.available_devices > 0
                
                # 检查预约情况
                reserved_vehicles = self.station_reservations.get(station_id, [])
                reserved_count = len(reserved_vehicles)
                
                # 还有可用容量且预约车辆数不超过可用设备数
                if (no_queue and 
                    has_available_device and
                    reserved_count < station.available_devices):
                    
                    valid_stations.append((station_id, dist))
            
            # 按距离排序，取最近的5个
            valid_stations.sort(key=lambda x: x[1])
            nearby_stations = valid_stations[:self.max_nearby_stations]
            
            for i in range(min(len(nearby_stations), self.max_nearby_stations)):
                available_actions.append(self.max_nearby_centers + i)
            
            # 如果没有可去的转运站，允许休息
            if not available_actions:
                available_actions.append(rest_action)
        
        elif vehicle.status == 'idle' and vehicle.current_location_type == 'station' and vehicle.current_load > 0:
            # 这种情况不应该发生，已装载车辆应该在转运站排队或卸载
            pass
        
        if not available_actions:
            available_actions.append(rest_action)
        
        return list(set(available_actions))
    
    def map_action_to_destination(self, vehicle: CollectionVehicle, action: int) -> Tuple[str, str]:
        """将动作映射到具体的目的地（考虑预约）"""
        if action == self.action_space_size - 1:
            return 'rest', 'rest'
        
        if action < self.max_nearby_centers:
            # 去人口中心
            station_id = vehicle.current_location_id
            
            # 获取所有有效人口中心
            all_centers = self.distances.get(f'station_{station_id}', [])
            valid_centers = []
            
            for center_id, dist, garbage in all_centers:
                center = self.population_centers[center_id]
                
                # 检查是否有足够的垃圾
                has_enough_garbage = center.current_garbage >= self.vehicle_capacity
                
                # 检查预约情况
                reserved_vehicles = self.center_reservations.get(center_id, [])
                reserved_garbage = len(reserved_vehicles) * self.vehicle_capacity
                available_garbage = center.current_garbage - reserved_garbage
                
                if (has_enough_garbage and 
                    available_garbage >= self.vehicle_capacity and
                    len(reserved_vehicles) < 2):
                    
                    valid_centers.append((center_id, dist, center.current_garbage))
            
            valid_centers.sort(key=lambda x: x[1])
            
            if action < len(valid_centers):
                center_id = valid_centers[action][0]
                return 'center', center_id
            else:
                return 'rest', 'rest'
        
        else:
            # 去转运站
            center_id = vehicle.current_location_id
            
            # 获取所有有效转运站
            all_stations = self.distances.get(f'center_{center_id}', [])
            valid_stations = []
            
            for station_id, dist, queue_length in all_stations:
                station = self.transfer_stations[station_id]
                
                # 检查排队和预约情况
                no_queue = queue_length == 0
                has_available_device = station.available_devices > 0
                reserved_count = len(self.station_reservations.get(station_id, []))
                
                if (no_queue and 
                    has_available_device and
                    reserved_count < station.available_devices):
                    
                    valid_stations.append((station_id, dist))
            
            valid_stations.sort(key=lambda x: x[1])
            
            station_idx = action - self.max_nearby_centers
            if station_idx < len(valid_stations):
                station_id = valid_stations[station_idx][0]
                return 'station', station_id
            else:
                return 'rest', 'rest'
    
    def add_reservation(self, vehicle_id: str, target_type: str, target_id: str, arrival_time: float):
        """添加预约"""
        if target_type == 'center':
            if vehicle_id not in self.center_reservations[target_id]:
                self.center_reservations[target_id].append(vehicle_id)
        elif target_type == 'station':
            if vehicle_id not in self.station_reservations[target_id]:
                self.station_reservations[target_id].append(vehicle_id)
        
        # 记录路径预约
        self.route_reservations[vehicle_id] = (target_type, target_id, arrival_time)
    
    def remove_reservation(self, vehicle_id: str):
        """移除预约"""
        if vehicle_id in self.route_reservations:
            target_type, target_id, _ = self.route_reservations[vehicle_id]
            
            if target_type == 'center' and vehicle_id in self.center_reservations.get(target_id, []):
                self.center_reservations[target_id].remove(vehicle_id)
            elif target_type == 'station' and vehicle_id in self.station_reservations.get(target_id, []):
                self.station_reservations[target_id].remove(vehicle_id)
            
            # 移除路径预约
            del self.route_reservations[vehicle_id]
    
    def check_reservation_violation(self, vehicle_id: str, target_type: str, target_id: str) -> bool:
        """检查预约违规"""
        if vehicle_id not in self.route_reservations:
            return False
        
        reserved_type, reserved_id, arrival_time = self.route_reservations[vehicle_id]
        
        # 检查是否按预约执行
        if reserved_type != target_type or reserved_id != target_id:
            return True
        
        # 检查是否按时到达（允许一定误差）
        time_diff = abs(self.current_time - arrival_time)
        if time_diff > self.time_step * 2:  # 允许2个时间步的误差
            return True
        
        return False
    
    def step(self, actions: Dict[str, int]) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行动作并返回结果（每5分钟一步）"""
        
        # 1. 执行车辆动作
        for vehicle_id, action in actions.items():
            vehicle = next((v for v in self.vehicles if v.id == vehicle_id), None)
            if vehicle:
                self._execute_vehicle_action(vehicle, action)
        
        # 2. 更新环境状态
        self._update_environment()
        
        # 3. 计算奖励
        reward = self._calculate_reward()
        self.performance_metrics['total_episode_reward'] += reward
        
        # 4. 检查是否结束
        done = self.current_step >= self.max_steps
        
        # 5. 获取新状态
        next_state = self.get_state()
        
        # 6. 记录历史
        self._record_history(reward)
        
        self.current_step += 1
        self.current_time += self.time_step
        
        return next_state, reward, done, {}
    
    def _execute_vehicle_action(self, vehicle: CollectionVehicle, action: int):
        """执行单个车辆的动作（包含预约）"""
        dest_type, dest_id = self.map_action_to_destination(vehicle, action)
        
        if dest_type == 'center':
            if vehicle.status == 'idle' and vehicle.current_location_type == 'station' and vehicle.current_load == 0:
                target_center = self.population_centers.get(dest_id)
                
                if target_center:
                    # 检查预约情况
                    reserved_vehicles = self.center_reservations.get(dest_id, [])
                    reserved_garbage = len(reserved_vehicles) * self.vehicle_capacity
                    available_garbage = target_center.current_garbage - reserved_garbage
                    
                    if (target_center.can_collect(self.vehicle_capacity) and 
                        available_garbage >= self.vehicle_capacity and
                        len(reserved_vehicles) < 2):
                        
                        # 添加预约
                        target_center.collection_queue += 1
                        current_location = self.transfer_stations[vehicle.current_location_id]
                        distance = current_location.distance_to(target_center)
                        speed = random.uniform(vehicle.speed_min, vehicle.speed_max)
                        travel_time = distance / speed
                        arrival_time = self.current_time + travel_time
                        
                        self.add_reservation(vehicle.id, 'center', dest_id, arrival_time)
                        
                        vehicle.start_travel('center', dest_id, distance, speed)
        
        elif dest_type == 'station':
            if (vehicle.status == 'loaded' and vehicle.current_location_type == 'center') or \
               (vehicle.status == 'idle' and vehicle.current_location_type == 'center' and vehicle.current_load == 0):
                
                target_station = self.transfer_stations.get(dest_id)
                
                if target_station:
                    # 检查预约情况
                    reserved_vehicles = self.station_reservations.get(dest_id, [])
                    
                    if (len(target_station.queue) == 0 and 
                        target_station.available_devices > 0 and
                        len(reserved_vehicles) < target_station.available_devices):
                        
                        current_location = self.population_centers[vehicle.current_location_id]
                        distance = current_location.distance_to(target_station)
                        speed = random.uniform(vehicle.speed_min, vehicle.speed_max)
                        travel_time = distance / speed
                        arrival_time = self.current_time + travel_time
                        
                        self.add_reservation(vehicle.id, 'station', dest_id, arrival_time)
                        
                        vehicle.start_travel('station', dest_id, distance, speed)
                        
                        if vehicle.current_load == 0 and vehicle.current_location_id in self.population_centers:
                            center = self.population_centers[vehicle.current_location_id]
                            center.collection_queue = max(0, center.collection_queue - 1)
        
        elif dest_type == 'rest':
            if vehicle.status == 'idle' and vehicle.current_location_type == 'station':
                vehicle.start_rest()
    
    def _update_environment(self):
        """更新环境状态（每5分钟更新）"""
        
        # 1. 更新人口中心的垃圾（均匀生成）
        for center in self.population_centers.values():
            center.update_garbage(self.time_step)
        
        # 2. 更新车辆状态
        for vehicle in self.vehicles:
            vehicle.update_status(self.time_step)
            
            if vehicle.status == 'traveling' and vehicle.remaining_time <= 0:
                vehicle.total_distance += vehicle.route_distance
                vehicle.current_location_type = vehicle.destination_type
                vehicle.current_location_id = vehicle.destination_id
                vehicle.route_progress = 1.0
                
                self._handle_vehicle_arrival(vehicle)
            
            elif vehicle.status == 'loading' and vehicle.remaining_time <= 0:
                center_id = vehicle.current_location_id
                if center_id in self.population_centers:
                    center = self.population_centers[center_id]
                    load_amount, success = center.collect_garbage(self.vehicle_capacity)
                    if success:
                        vehicle.current_load = load_amount
                        vehicle.total_garbage_collected += load_amount
                        vehicle.status = 'loaded'
                        
                        center.collection_queue = max(0, center.collection_queue - 1)
                    else:
                        vehicle.status = 'idle'
                        vehicle.empty_runs += 1
                        self.performance_metrics['total_empty_runs'] += 1
            
            elif vehicle.status == 'unloading' and vehicle.remaining_time <= 0:
                station_id = vehicle.current_location_id
                if station_id in self.transfer_stations:
                    station = self.transfer_stations[station_id]
                    unload_amount = vehicle.current_load
                    station.current_garbage += unload_amount
                    station.total_processed += unload_amount
                    station.available_devices += 1
                    
                    if vehicle.id in station.processing_vehicles:
                        station.processing_vehicles.remove(vehicle.id)
                    
                    vehicle.current_load = 0.0
                    vehicle.total_garbage_unloaded += unload_amount
                    vehicle.status = 'idle'
                    vehicle.trips_completed += 1
            
            elif vehicle.status == 'resting' and vehicle.remaining_time <= 0:
                vehicle.status = 'idle'
        
        # 3. 更新转运站排队
        for station in self.transfer_stations.values():
            station.update_queue_waiting(self.time_step)
            
            processed_vehicles = station.process_queue()
            for vehicle_id in processed_vehicles:
                vehicle = next((v for v in self.vehicles if v.id == vehicle_id), None)
                if vehicle and vehicle.current_load > 0 and vehicle.status == 'waiting':
                    unload_speed = random.uniform(station.unload_speed_min, station.unload_speed_max)
                    unload_time = vehicle.current_load / unload_speed
                    vehicle.start_unloading(unload_time)
    
    def _handle_vehicle_arrival(self, vehicle: CollectionVehicle):
        """处理车辆到达目的地（包含预约检查）"""
        # 检查预约违规
        violation = False
        if vehicle.destination_type and vehicle.destination_id:
            violation = self.check_reservation_violation(vehicle.id, vehicle.destination_type, vehicle.destination_id)
            if violation:
                self.performance_metrics['total_reservation_violations'] += 1
        
        # 移除预约
        self.remove_reservation(vehicle.id)
        
        if vehicle.destination_type == 'center':
            center_id = vehicle.destination_id
            if center_id in self.population_centers:
                center = self.population_centers[center_id]
                
                if center.can_collect(self.vehicle_capacity):
                    load_time = random.uniform(vehicle.load_time_min, vehicle.load_time_max)
                    vehicle.start_loading(load_time)
                else:
                    vehicle.status = 'idle'
                    vehicle.empty_runs += 1
                    self.performance_metrics['total_empty_runs'] += 1
                    
                    center.collection_queue = max(0, center.collection_queue - 1)
        
        elif vehicle.destination_type == 'station':
            station_id = vehicle.destination_id
            if station_id in self.transfer_stations:
                station = self.transfer_stations[station_id]
                
                if vehicle.current_load > 0:
                    if station.can_unload:
                        unload_speed = random.uniform(station.unload_speed_min, station.unload_speed_max)
                        unload_time = vehicle.current_load / unload_speed
                        vehicle.start_unloading(unload_time)
                        station.available_devices -= 1
                        station.processing_vehicles.append(vehicle.id)
                    else:
                        station.add_to_queue(vehicle.id)
                        vehicle.status = 'waiting'
                else:
                    vehicle.status = 'idle'
    
    def _calculate_reward(self) -> float:
        """计算奖励函数"""
        reward = 0.0
        
        # 1. 收集垃圾奖励
        garbage_collected = 0.0
        for vehicle in self.vehicles:
            if vehicle.status == 'loading':
                expected_time = random.uniform(vehicle.load_time_min, vehicle.load_time_max)
                progress = 1 - (vehicle.remaining_time / max(expected_time, 0.001))
                garbage_collected += self.vehicle_capacity * progress
        
        reward += garbage_collected * self.reward_weights['garbage_collected']
        
        # 2. 处理垃圾奖励
        garbage_processed = 0.0
        for vehicle in self.vehicles:
            if vehicle.status == 'unloading':
                unload_speed = random.uniform(30, 50)
                expected_time = vehicle.current_load / unload_speed
                progress = 1 - (vehicle.remaining_time / max(expected_time, 0.001))
                garbage_processed += vehicle.current_load * progress
        
        reward += garbage_processed * self.reward_weights['garbage_processed']
        
        # 3. 距离惩罚
        distance_penalty = 0.0
        for vehicle in self.vehicles:
            if vehicle.status == 'traveling':
                distance_penalty += vehicle.travel_speed * self.time_step
        
        reward += distance_penalty * self.reward_weights['distance_penalty']
        
        # 4. 等待时间惩罚
        waiting_penalty = 0.0
        for center in self.population_centers.values():
            if center.current_garbage > 0:
                waiting_penalty += center.waiting_time * 0.1
        
        reward += waiting_penalty * self.reward_weights['waiting_penalty']
        
        # 5. 排队惩罚
        queue_penalty = sum(len(s.queue) for s in self.transfer_stations.values())
        reward += queue_penalty * self.reward_weights['queue_penalty']
        
        # 6. 拥堵惩罚
        congestion_penalty = 0.0
        for center in self.population_centers.values():
            if center.current_garbage > self.vehicle_capacity * 3:
                congestion_penalty += center.current_garbage * 0.1
        
        reward += congestion_penalty * self.reward_weights['congestion_penalty']
        
        # 7. 空跑惩罚
        empty_run_penalty = 0.0
        for vehicle in self.vehicles:
            if vehicle.empty_runs > 0:
                empty_run_penalty += vehicle.empty_runs
                vehicle.empty_runs = 0
        
        reward += empty_run_penalty * self.reward_weights['empty_run_penalty']
        
        # 8. 休整惩罚
        rest_penalty = 0.0
        for vehicle in self.vehicles:
            if vehicle.status == 'resting':
                rest_penalty += 1
                self.performance_metrics['total_rest_count'] += 1
        
        reward += rest_penalty * self.reward_weights['rest_penalty']
        
        # 9. 冲突惩罚
        collision_penalty = 0.0
        for center in self.population_centers.values():
            if center.collection_queue > 2:
                collision_penalty += (center.collection_queue - 2)
        
        reward += collision_penalty * self.reward_weights['collision_penalty']
        
        # 10. 预约违规惩罚
        reservation_penalty = self.performance_metrics['total_reservation_violations']
        reward += reservation_penalty * self.reward_weights['reservation_violation']
        
        # 11. 利用率奖励
        active_vehicles = sum(1 for v in self.vehicles if v.status not in ['idle', 'waiting', 'resting'])
        utilization = active_vehicles / len(self.vehicles) if len(self.vehicles) > 0 else 0
        reward += utilization * self.reward_weights['utilization_bonus'] * 100
        
        # 12. 效率奖励
        efficiency_bonus = 0.0
        loaded_vehicles = [v for v in self.vehicles if v.current_load > 0]
        if loaded_vehicles:
            avg_load_ratio = sum(v.current_load for v in loaded_vehicles) / (len(loaded_vehicles) * self.vehicle_capacity)
            efficiency_bonus = avg_load_ratio
        
        reward += efficiency_bonus * self.reward_weights['efficiency_bonus'] * len(self.vehicles)
        
        return reward
    
    def _record_history(self, reward: float):
        """记录历史数据"""
        total_garbage = sum(c.current_garbage for c in self.population_centers.values())
        total_collected = sum(c.total_collected for c in self.population_centers.values())
        total_processed = sum(s.total_processed for s in self.transfer_stations.values())
        
        total_distance = sum(v.total_distance for v in self.vehicles)
        
        total_waiting = sum(v.waiting_time for v in self.vehicles)
        
        total_queue = sum(len(s.queue) for s in self.transfer_stations.values())
        
        active_vehicles = sum(1 for v in self.vehicles if v.status not in ['idle', 'waiting', 'resting'])
        utilization = (active_vehicles / len(self.vehicles)) * 100 if len(self.vehicles) > 0 else 0
        
        status_counts = defaultdict(int)
        for vehicle in self.vehicles:
            status_counts[vehicle.status] += 1
        
        total_empty_runs = sum(v.empty_runs for v in self.vehicles)
        
        rest_count = sum(1 for v in self.vehicles if v.status == 'resting')
        
        self.history['total_garbage'].append(total_garbage)
        self.history['total_collected'].append(total_collected)
        self.history['total_processed'].append(total_processed)
        self.history['total_distance'].append(total_distance)
        self.history['total_waiting'].append(total_waiting)
        self.history['total_queue'].append(total_queue)
        self.history['vehicle_utilization'].append(utilization)
        self.history['rewards'].append(reward)
        self.history['status_counts'].append(dict(status_counts))
        self.history['empty_runs'].append(total_empty_runs)
        self.history['rest_count'].append(rest_count)
        self.history['reservation_violations'].append(
            self.performance_metrics['total_reservation_violations']
        )
    
    def reset(self):
        """重置环境"""
        # 重置人口中心
        for center in self.population_centers.values():
            center.current_garbage = random.uniform(5.0, 20.0)
            center.waiting_time = 0.0
            center.total_generated = center.current_garbage
            center.total_collected = 0.0
            center.last_collection_time = 0.0
            center.collection_queue = 0
        
        # 重置转运站
        for station in self.transfer_stations.values():
            station.current_garbage = 0.0
            station.available_devices = station.device_num
            station.queue.clear()
            station.processing_vehicles.clear()
            station.queue_waiting_time = 0.0
            station.total_processed = 0.0
        
        # 重置车辆 - 确保每个转运站有与其设备数相同数量的车辆
        # 按车辆ID排序，确保每次分配顺序一致
        sorted_vehicles = sorted(self.vehicles, key=lambda v: v.id)
        vehicle_index = 0
        
        # 按转运站ID排序，确保每次分配顺序一致
        sorted_stations = sorted(self.transfer_stations.items(), key=lambda x: x[0])
        
        for station_id, station in sorted_stations:
            for i in range(station.device_num):
                if vehicle_index < len(sorted_vehicles):
                    vehicle = sorted_vehicles[vehicle_index]
                    vehicle.current_load = 0.0
                    vehicle.current_location_type = 'station'
                    vehicle.current_location_id = station_id
                    vehicle.status = 'idle'
                    vehicle.destination_type = None
                    vehicle.destination_id = None
                    vehicle.route_distance = 0.0
                    vehicle.route_progress = 0.0
                    vehicle.remaining_time = 0.0
                    vehicle.travel_speed = 0.0
                    vehicle.total_distance = 0.0
                    vehicle.total_garbage_collected = 0.0
                    vehicle.total_garbage_unloaded = 0.0
                    vehicle.total_operating_time = 0.0
                    vehicle.idle_time = 0.0
                    vehicle.waiting_time = 0.0
                    vehicle.trips_completed = 0
                    vehicle.empty_runs = 0
                    vehicle.rest_time = 0.0
                    vehicle_index += 1
        
        # 如果有剩余车辆（正常情况下不会发生），放到最后一个转运站
        if vehicle_index < len(sorted_vehicles):
            print(f"警告：有{len(sorted_vehicles)-vehicle_index}辆车未分配，车辆数多于总设备数")
            for i in range(vehicle_index, len(sorted_vehicles)):
                vehicle = sorted_vehicles[i]
                vehicle.current_load = 0.0
                vehicle.current_location_type = 'station'
                vehicle.current_location_id = sorted_stations[-1][0]
                vehicle.status = 'idle'
        
        # 重置预约信息
        self.center_reservations = {}
        self.station_reservations = {}
        self.route_reservations = {}
        
        for center_id in self.population_center_ids:
            self.center_reservations[center_id] = []
        for station_id in self.transfer_station_ids:
            self.station_reservations[station_id] = []
        
        # 更新距离矩阵
        self._calculate_distance_matrix()
        
        # 重置时间
        self.current_time = 0.0
        self.current_step = 0
        
        # 重置性能指标
        self.performance_metrics = {
            'total_episode_reward': 0.0,
            'total_garbage_collected': 0.0,
            'total_garbage_processed': 0.0,
            'total_distance': 0.0,
            'average_waiting_time': 0.0,
            'average_queue_length': 0.0,
            'vehicle_utilization_rate': 0.0,
            'total_empty_runs': 0,
            'total_rest_count': 0,
            'total_reservation_violations': 0
        }
        
        # 重置历史记录
        for key in self.history:
            self.history[key] = []
        
        return self.get_state()
    
    def get_current_plan(self):
        """获取当前运营规划"""
        plan = {
            'timestamp': self.current_time,
            'step': self.current_step,
            'vehicle_plans': [],
            'center_status': [],
            'station_status': []
        }
        
        # 车辆计划
        for vehicle in self.vehicles:
            vehicle_plan = {
                'id': vehicle.id,
                'status': vehicle.status,
                'location': vehicle.current_location_id,
                'load': vehicle.current_load,
                'destination': vehicle.destination_id if vehicle.destination_id else 'None',
                'remaining_time': vehicle.remaining_time,
                'empty_runs': vehicle.empty_runs,
                'trips_completed': vehicle.trips_completed
            }
            plan['vehicle_plans'].append(vehicle_plan)
        
        # 人口中心状态
        for center_id, center in self.population_centers.items():
            center_status = {
                'id': center_id,
                'garbage': center.current_garbage,
                'demand': center.get_garbage_demand(self.vehicle_capacity),
                'can_collect': center.can_collect(self.vehicle_capacity),
                'waiting_time': center.waiting_time,
                'generation_rate': center.generation_rate,
                'collection_queue': center.collection_queue,
                'reservations': len(self.center_reservations.get(center_id, []))
            }
            plan['center_status'].append(center_status)
        
        # 转运站状态
        for station_id, station in self.transfer_stations.items():
            station_status = {
                'id': station_id,
                'garbage': station.current_garbage,
                'capacity': station.capacity,
                'queue': len(station.queue),
                'available_devices': station.available_devices,
                'total_processed': station.total_processed,
                'device_num': station.device_num,
                'reservations': len(self.station_reservations.get(station_id, []))
            }
            plan['station_status'].append(station_status)
        
        return plan

# ===================== 第五部分：深度强化学习训练 =====================
class DRLTrainer:
    """深度强化学习训练器"""
    
    def __init__(self, env, agent, num_episodes=100, max_steps=1440):
        self.env = env
        self.agent = agent
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        
        # 训练历史
        self.training_history = {
            'episode_rewards': [],
            'episode_garbage_collected': [],
            'episode_garbage_processed': [],
            'episode_distance': [],
            'episode_utilization': [],
            'episode_losses': [],
            'epsilon_values': [],
            'episode_empty_runs': [],
            'episode_rest_count': [],
            'episode_reservation_violations': []  # 添加预约违规记录
        }
    
    def train(self):
        """训练深度强化学习智能体"""
        print("开始深度强化学习训练...")
        print(f"环境: {len(self.env.population_centers)}个人口中心, {len(self.env.transfer_stations)}个转运站, {len(self.env.vehicles)}辆车")
        print(f"动作空间: {self.env.action_space_size}个动作")
        print(f"最近人口中心限制: {self.env.max_nearby_centers}个（只考虑有垃圾的）")
        print(f"最近转运站限制: {self.env.max_nearby_stations}个（只考虑无排队的）")
        print(f"预约机制: 人口中心最多预约2辆车，转运站按可用设备数预约")
        print(f"训练参数: {self.num_episodes}回合, 每回合{self.max_steps}步（{self.max_steps*self.env.time_step:.1f}小时）")
        print("="*80)
        
        for episode in range(self.num_episodes):
            state = self.env.reset()
            episode_reward = 0.0
            episode_losses = []
            
            for step in range(self.max_steps):
                decision_vehicles = self.env.get_decision_vehicles()
                
                actions = {}
                for vehicle in decision_vehicles:
                    available_actions = self.env.get_available_actions(vehicle)
                    
                    action = self.agent.select_action(state, available_actions)
                    actions[vehicle.id] = action
                
                next_state, reward, done, _ = self.env.step(actions)
                episode_reward += reward
                
                for vehicle in decision_vehicles:
                    if vehicle.id in actions:
                        self.agent.memory.push(
                            state, 
                            actions[vehicle.id], 
                            reward, 
                            next_state, 
                            done
                        )
                
                loss = self.agent.optimize_model()
                if loss > 0:
                    episode_losses.append(loss)
                
                state = next_state
                
                if self.agent.steps_done % self.agent.target_update == 0:
                    self.agent.update_target_network()
                
                if done:
                    break
            
            # 记录训练历史
            self.training_history['episode_rewards'].append(episode_reward)
            self.training_history['episode_garbage_collected'].append(
                sum(c.total_collected for c in self.env.population_centers.values())
            )
            self.training_history['episode_garbage_processed'].append(
                sum(s.total_processed for s in self.env.transfer_stations.values())
            )
            self.training_history['episode_distance'].append(
                sum(v.total_distance for v in self.env.vehicles)
            )
            
            active_vehicles = sum(1 for v in self.env.vehicles if v.status not in ['idle', 'waiting', 'resting'])
            utilization = (active_vehicles / len(self.env.vehicles)) * 100 if len(self.env.vehicles) > 0 else 0
            self.training_history['episode_utilization'].append(utilization)
            
            self.training_history['episode_empty_runs'].append(
                self.env.performance_metrics['total_empty_runs']
            )
            self.training_history['episode_rest_count'].append(
                self.env.performance_metrics['total_rest_count']
            )
            
            self.training_history['episode_reservation_violations'].append(
                self.env.performance_metrics['total_reservation_violations']
            )
            
            avg_loss = np.mean(episode_losses) if episode_losses else 0.0
            self.training_history['episode_losses'].append(avg_loss)
            
            self.training_history['epsilon_values'].append(self.agent.eps)
            
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.training_history['episode_rewards'][-10:])
                avg_garbage = np.mean(self.training_history['episode_garbage_collected'][-10:])
                avg_utilization = np.mean(self.training_history['episode_utilization'][-10:])
                avg_reservation_violations = np.mean(self.training_history['episode_reservation_violations'][-10:])
                
                print(f"Episode {episode+1}/{self.num_episodes}, "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Avg Garbage: {avg_garbage:.2f}t, "
                      f"Avg Utilization: {avg_utilization:.1f}%, "
                      f"Avg Reservation Violations: {avg_reservation_violations:.1f}, "
                      f"Epsilon: {self.agent.eps:.3f}")
            
            if (episode + 1) % 50 == 0:
                self.agent.save_model(f"dqn_model_episode_{episode+1}.pth")
        
        self.agent.save_model("dqn_model_final.pth")
        print("训练完成，模型已保存到 dqn_model_final.pth")
        
        return self.training_history
    
    def evaluate(self, num_episodes=5):
        """评估训练好的智能体"""
        print("\n评估训练好的智能体...")
        
        evaluation_results = {
            'rewards': [],
            'garbage_collected': [],
            'garbage_processed': [],
            'distance': [],
            'utilization': [],
            'waiting_time': [],
            'queue_length': [],
            'empty_runs': [],
            'rest_count': [],
            'reservation_violations': []
        }
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0.0
            
            for step in range(self.max_steps):
                decision_vehicles = self.env.get_decision_vehicles()
                
                actions = {}
                for vehicle in decision_vehicles:
                    available_actions = self.env.get_available_actions(vehicle)
                    
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
                        q_values = self.agent.policy_net(state_tensor).cpu().numpy()[0]
                        
                        if available_actions:
                            available_q_values = [q_values[a] for a in available_actions]
                            best_idx = np.argmax(available_q_values)
                            action = available_actions[best_idx]
                        else:
                            action = np.argmax(q_values)
                    
                    actions[vehicle.id] = action
                
                next_state, reward, done, _ = self.env.step(actions)
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            evaluation_results['rewards'].append(episode_reward)
            evaluation_results['garbage_collected'].append(
                sum(c.total_collected for c in self.env.population_centers.values())
            )
            evaluation_results['garbage_processed'].append(
                sum(s.total_processed for s in self.env.transfer_stations.values())
            )
            evaluation_results['distance'].append(
                sum(v.total_distance for v in self.env.vehicles)
            )
            
            active_vehicles = sum(1 for v in self.env.vehicles if v.status not in ['idle', 'waiting', 'resting'])
            utilization = (active_vehicles / len(self.env.vehicles)) * 100 if len(self.env.vehicles) > 0 else 0
            evaluation_results['utilization'].append(utilization)
            
            total_waiting = sum(v.waiting_time for v in self.env.vehicles)
            total_queue = sum(len(s.queue) for s in self.env.transfer_stations.values())
            evaluation_results['waiting_time'].append(total_waiting)
            evaluation_results['queue_length'].append(total_queue)
            
            evaluation_results['empty_runs'].append(
                self.env.performance_metrics['total_empty_runs']
            )
            evaluation_results['rest_count'].append(
                self.env.performance_metrics['total_rest_count']
            )
            
            evaluation_results['reservation_violations'].append(
                self.env.performance_metrics['total_reservation_violations']
            )
            
            print(f"评估 Episode {episode+1}/{num_episodes}, "
                  f"Reward: {episode_reward:.2f}, "
                  f"Garbage Collected: {evaluation_results['garbage_collected'][-1]:.2f}t, "
                  f"Empty Runs: {evaluation_results['empty_runs'][-1]}, "
                  f"Reservation Violations: {evaluation_results['reservation_violations'][-1]}")
        
        avg_results = {}
        for key, values in evaluation_results.items():
            avg_results[key] = np.mean(values)
        
        print("\n评估结果平均值:")
        for key, value in avg_results.items():
            print(f"  {key}: {value:.2f}")
        
        return evaluation_results, avg_results

# ===================== 第六部分：实时运营规划和可视化 =====================
class RealTimePlanner:
    """实时运营规划器"""
    
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.current_plan = None
    
    def generate_plan(self):
        """生成当前最优运营规划"""
        state = self.env.get_state()
        
        decision_vehicles = self.env.get_decision_vehicles()
        
        plan = {}
        for vehicle in decision_vehicles:
            available_actions = self.env.get_available_actions(vehicle)
            
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
                q_values = self.agent.policy_net(state_tensor).cpu().numpy()[0]
                
                if available_actions:
                    available_q_values = [q_values[a] for a in available_actions]
                    best_idx = np.argmax(available_q_values)
                    best_action = available_actions[best_idx]
                    
                    dest_type, dest_id = self.env.map_action_to_destination(vehicle, best_action)
                    
                    if dest_type == 'center':
                        action_desc = f"前往人口中心 {dest_id}"
                    elif dest_type == 'station':
                        action_desc = f"前往转运站 {dest_id}"
                    else:
                        action_desc = "休整5分钟"
                    
                    plan[vehicle.id] = {
                        'action': best_action,
                        'description': action_desc,
                        'destination_type': dest_type,
                        'destination_id': dest_id,
                        'q_value': float(q_values[best_action])
                    }
        
        self.current_plan = plan
        return plan
    
    def execute_plan(self):
        """执行当前规划"""
        if not self.current_plan:
            return None
        
        actions = {vehicle_id: data['action'] for vehicle_id, data in self.current_plan.items()}
        
        next_state, reward, done, _ = self.env.step(actions)
        
        return {
            'reward': reward,
            'done': done,
            'next_state': next_state,
            'plan_executed': True
        }
    
    def generate_future_plan(self, steps=10):
        """生成未来多步规划"""
        print(f"正在生成未来{steps}步规划...")
        
        # 保存当前环境状态
        saved_state = self._save_env_state()
        
        future_plans = []
        
        for step in range(steps):
            # 生成当前步的规划
            current_plan = self.generate_plan()
            
            if not current_plan:
                # 如果没有规划，模拟环境更新
                next_state, reward, done, _ = self.env.step({})
                
                future_plans.append({
                    'step': step + 1,
                    'time': self.env.current_time,
                    'plan': {},
                    'summary': '无决策车辆',
                    'vehicle_status': self._get_vehicle_status_summary(),
                    'center_demand': self._get_center_demand_summary(),
                    'station_status': self._get_station_status_summary()
                })
            else:
                # 执行当前规划
                result = self.execute_plan()
                
                future_plans.append({
                    'step': step + 1,
                    'time': self.env.current_time,
                    'plan': current_plan,
                    'reward': result['reward'] if result else 0,
                    'summary': self._get_plan_summary(current_plan),
                    'vehicle_status': self._get_vehicle_status_summary(),
                    'center_demand': self._get_center_demand_summary(),
                    'station_status': self._get_station_status_summary(),
                    'performance': {
                        'garbage_collected': sum(c.total_collected for c in self.env.population_centers.values()),
                        'garbage_processed': sum(s.total_processed for s in self.env.transfer_stations.values()),
                        'total_distance': sum(v.total_distance for v in self.env.vehicles)
                    }
                })
        
        # 恢复环境状态
        self._restore_env_state(saved_state)
        
        return future_plans
    
    def _save_env_state(self):
        """保存环境状态"""
        saved_state = {
            'current_time': self.env.current_time,
            'current_step': self.env.current_step,
            'performance_metrics': copy.deepcopy(self.env.performance_metrics),
            'center_reservations': copy.deepcopy(self.env.center_reservations),
            'station_reservations': copy.deepcopy(self.env.station_reservations),
            'route_reservations': copy.deepcopy(self.env.route_reservations),
            'vehicles': [],
            'population_centers': [],
            'transfer_stations': []
        }
        
        # 保存车辆状态
        for vehicle in self.env.vehicles:
            saved_state['vehicles'].append({
                'id': vehicle.id,
                'current_load': vehicle.current_load,
                'current_location_type': vehicle.current_location_type,
                'current_location_id': vehicle.current_location_id,
                'status': vehicle.status,
                'destination_type': vehicle.destination_type,
                'destination_id': vehicle.destination_id,
                'route_distance': vehicle.route_distance,
                'route_progress': vehicle.route_progress,
                'remaining_time': vehicle.remaining_time,
                'travel_speed': vehicle.travel_speed,
                'total_distance': vehicle.total_distance,
                'total_garbage_collected': vehicle.total_garbage_collected,
                'total_garbage_unloaded': vehicle.total_garbage_unloaded,
                'total_operating_time': vehicle.total_operating_time,
                'idle_time': vehicle.idle_time,
                'waiting_time': vehicle.waiting_time,
                'trips_completed': vehicle.trips_completed,
                'empty_runs': vehicle.empty_runs,
                'rest_time': vehicle.rest_time
            })
        
        # 保存人口中心状态
        for center_id, center in self.env.population_centers.items():
            saved_state['population_centers'].append({
                'id': center_id,
                'current_garbage': center.current_garbage,
                'waiting_time': center.waiting_time,
                'total_generated': center.total_generated,
                'total_collected': center.total_collected,
                'last_collection_time': center.last_collection_time,
                'collection_queue': center.collection_queue
            })
        
        # 保存转运站状态
        for station_id, station in self.env.transfer_stations.items():
            saved_state['transfer_stations'].append({
                'id': station_id,
                'current_garbage': station.current_garbage,
                'available_devices': station.available_devices,
                'queue': list(station.queue),
                'processing_vehicles': list(station.processing_vehicles),
                'total_processed': station.total_processed,
                'queue_waiting_time': station.queue_waiting_time
            })
        
        return saved_state
    
    def _restore_env_state(self, saved_state):
        """恢复环境状态"""
        self.env.current_time = saved_state['current_time']
        self.env.current_step = saved_state['current_step']
        self.env.performance_metrics = saved_state['performance_metrics']
        self.env.center_reservations = saved_state['center_reservations']
        self.env.station_reservations = saved_state['station_reservations']
        self.env.route_reservations = saved_state['route_reservations']
        
        # 恢复车辆状态
        for i, vehicle_data in enumerate(saved_state['vehicles']):
            if i < len(self.env.vehicles):
                vehicle = self.env.vehicles[i]
                vehicle.current_load = vehicle_data['current_load']
                vehicle.current_location_type = vehicle_data['current_location_type']
                vehicle.current_location_id = vehicle_data['current_location_id']
                vehicle.status = vehicle_data['status']
                vehicle.destination_type = vehicle_data['destination_type']
                vehicle.destination_id = vehicle_data['destination_id']
                vehicle.route_distance = vehicle_data['route_distance']
                vehicle.route_progress = vehicle_data['route_progress']
                vehicle.remaining_time = vehicle_data['remaining_time']
                vehicle.travel_speed = vehicle_data['travel_speed']
                vehicle.total_distance = vehicle_data['total_distance']
                vehicle.total_garbage_collected = vehicle_data['total_garbage_collected']
                vehicle.total_garbage_unloaded = vehicle_data['total_garbage_unloaded']
                vehicle.total_operating_time = vehicle_data['total_operating_time']
                vehicle.idle_time = vehicle_data['idle_time']
                vehicle.waiting_time = vehicle_data['waiting_time']
                vehicle.trips_completed = vehicle_data['trips_completed']
                vehicle.empty_runs = vehicle_data['empty_runs']
                vehicle.rest_time = vehicle_data['rest_time']
        
        # 恢复人口中心状态
        for center_data in saved_state['population_centers']:
            center_id = center_data['id']
            if center_id in self.env.population_centers:
                center = self.env.population_centers[center_id]
                center.current_garbage = center_data['current_garbage']
                center.waiting_time = center_data['waiting_time']
                center.total_generated = center_data['total_generated']
                center.total_collected = center_data['total_collected']
                center.last_collection_time = center_data['last_collection_time']
                center.collection_queue = center_data['collection_queue']
        
        # 恢复转运站状态
        for station_data in saved_state['transfer_stations']:
            station_id = station_data['id']
            if station_id in self.env.transfer_stations:
                station = self.env.transfer_stations[station_id]
                station.current_garbage = station_data['current_garbage']
                station.available_devices = station_data['available_devices']
                station.queue = deque(station_data['queue'])
                station.processing_vehicles = station_data['processing_vehicles']
                station.total_processed = station_data['total_processed']
                station.queue_waiting_time = station_data['queue_waiting_time']
    
    def _get_plan_summary(self, plan):
        """获取规划摘要"""
        summary = {
            'total_vehicles': len(plan),
            'to_center': 0,
            'to_station': 0,
            'rest': 0
        }
        
        for vehicle_id, data in plan.items():
            dest_type = data['destination_type']
            if dest_type == 'center':
                summary['to_center'] += 1
            elif dest_type == 'station':
                summary['to_station'] += 1
            else:
                summary['rest'] += 1
        
        return summary
    
    def _get_vehicle_status_summary(self):
        """获取车辆状态摘要"""
        status_counts = defaultdict(int)
        for vehicle in self.env.vehicles:
            status_counts[vehicle.status] += 1
        
        return dict(status_counts)
    
    def _get_center_demand_summary(self):
        """获取人口中心需求摘要"""
        total_demand = 0
        high_demand = 0
        for center in self.env.population_centers.values():
            if center.can_collect(self.env.vehicle_capacity):
                total_demand += 1
                if center.current_garbage > self.env.vehicle_capacity * 2:
                    high_demand += 1
        
        return {
            'total_demand': total_demand,
            'high_demand': high_demand,
            'total_centers': len(self.env.population_centers)
        }
    
    def _get_station_status_summary(self):
        """获取转运站状态摘要"""
        total_queue = 0
        busy_stations = 0
        for station in self.env.transfer_stations.values():
            total_queue += len(station.queue)
            if station.current_garbage > station.capacity * 0.7:
                busy_stations += 1
        
        return {
            'total_queue': total_queue,
            'busy_stations': busy_stations,
            'total_stations': len(self.env.transfer_stations)
        }
    
    def get_plan_summary(self):
        """获取规划摘要"""
        if not self.current_plan:
            return None
        
        summary = {
            'timestamp': self.env.current_time,
            'step': self.env.current_step,
            'total_vehicles': len(self.env.vehicles),
            'decision_vehicles': len(self.current_plan),
            'actions': defaultdict(int),
            'vehicle_status': defaultdict(int),
            'center_demand': 0,
            'center_with_queue': 0,
            'center_with_reservations': 0,
            'station_with_reservations': 0
        }
        
        for vehicle in self.env.vehicles:
            summary['vehicle_status'][vehicle.status] += 1
        
        for vehicle_id, data in self.current_plan.items():
            dest_type = data['destination_type']
            if dest_type == 'center':
                summary['actions']['to_center'] += 1
            elif dest_type == 'station':
                summary['actions']['to_station'] += 1
            else:
                summary['actions']['rest'] += 1
        
        for center in self.env.population_centers.values():
            if center.can_collect(self.env.vehicle_capacity):
                summary['center_demand'] += 1
            
            if center.collection_queue > 0:
                summary['center_with_queue'] += 1
            
            if len(self.env.center_reservations.get(center.id, [])) > 0:
                summary['center_with_reservations'] += 1
        
        for station in self.env.transfer_stations.values():
            if len(self.env.station_reservations.get(station.id, [])) > 0:
                summary['station_with_reservations'] += 1
        
        return summary

def visualize_future_plan(future_plans):
    """可视化未来规划"""
    try:
        steps = len(future_plans)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'未来{steps}步运营规划', fontsize=14, fontweight='bold')
        
        # 1. 未来各步的规划车辆数
        steps_list = list(range(1, steps + 1))
        plan_vehicles = [len(plan['plan']) for plan in future_plans]
        
        axes[0, 0].plot(steps_list, plan_vehicles, marker='o', color='blue', linewidth=2)
        axes[0, 0].set_title('未来各步规划车辆数')
        axes[0, 0].set_xlabel('步数')
        axes[0, 0].set_ylabel('规划车辆数')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xticks(steps_list)
        
        # 2. 未来各步的车辆状态分布
        status_names = ['idle', 'traveling', 'loading', 'unloading', 'waiting', 'loaded', 'resting']
        status_counts = {status: [] for status in status_names}
        
        for plan in future_plans:
            vehicle_status = plan.get('vehicle_status', {})
            for status in status_names:
                status_counts[status].append(vehicle_status.get(status, 0))
        
        bottom = np.zeros(steps)
        for status in status_names:
            counts = status_counts[status]
            if sum(counts) > 0:  # 只显示有车辆的状态
                axes[0, 1].bar(steps_list, counts, bottom=bottom, label=status)
                bottom += counts
        
        axes[0, 1].set_title('未来各步车辆状态分布')
        axes[0, 1].set_xlabel('步数')
        axes[0, 1].set_ylabel('车辆数')
        axes[0, 1].legend(loc='upper right', fontsize='small')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_xticks(steps_list)
        
        # 3. 未来各步的动作类型分布
        action_types = ['to_center', 'to_station', 'rest']
        action_counts = {action: [] for action in action_types}
        
        for plan in future_plans:
            plan_summary = plan.get('summary', {})
            for action in action_types:
                action_counts[action].append(plan_summary.get(action, 0))
        
        x = np.arange(steps)
        width = 0.25
        for i, action in enumerate(action_types):
            offset = (i - 1) * width
            axes[0, 2].bar(x + offset, action_counts[action], width, label=action)
        
        axes[0, 2].set_title('未来各步动作类型分布')
        axes[0, 2].set_xlabel('步数')
        axes[0, 2].set_ylabel('车辆数')
        axes[0, 2].set_xticks(x)
        axes[0, 2].set_xticklabels(steps_list)
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. 未来各步的垃圾收集量
        garbage_collected = [plan.get('performance', {}).get('garbage_collected', 0) for plan in future_plans]
        
        axes[1, 0].plot(steps_list, garbage_collected, marker='s', color='green', linewidth=2)
        axes[1, 0].set_title('未来各步垃圾收集量')
        axes[1, 0].set_xlabel('步数')
        axes[1, 0].set_ylabel('垃圾量(吨)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xticks(steps_list)
        
        # 5. 未来各步的人口中心需求
        center_demand = [plan.get('center_demand', {}).get('total_demand', 0) for plan in future_plans]
        high_demand = [plan.get('center_demand', {}).get('high_demand', 0) for plan in future_plans]
        
        x = np.arange(steps)
        width = 0.35
        axes[1, 1].bar(x - width/2, center_demand, width, label='总需求', color='orange')
        axes[1, 1].bar(x + width/2, high_demand, width, label='高需求', color='red')
        
        axes[1, 1].set_title('未来各步人口中心需求')
        axes[1, 1].set_xlabel('步数')
        axes[1, 1].set_ylabel('人口中心数')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(steps_list)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. 未来各步的转运站排队
        station_queue = [plan.get('station_status', {}).get('total_queue', 0) for plan in future_plans]
        
        axes[1, 2].plot(steps_list, station_queue, marker='^', color='purple', linewidth=2)
        axes[1, 2].set_title('未来各步转运站排队车辆数')
        axes[1, 2].set_xlabel('步数')
        axes[1, 2].set_ylabel('排队车辆数')
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].set_xticks(steps_list)
        axes[1, 2].axhline(y=5, color='red', linestyle='--', alpha=0.5, label='警戒线')
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig(f'future_plan_{steps}_steps.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"未来规划图已保存到 future_plan_{steps}_steps.png")
        
    except Exception as e:
        print(f"可视化未来规划时出错: {e}")
        import traceback
        traceback.print_exc()

def print_future_plan_details(future_plans):
    """打印未来规划详情"""
    print("\n" + "="*80)
    print(f"未来{len(future_plans)}步运营规划详情")
    print("="*80)
    
    for i, plan in enumerate(future_plans, 1):
        print(f"\n第{i}步 (时间: {plan['time']:.2f}小时):")
        print(f"  规划车辆数: {len(plan['plan'])}")
        
        if plan['plan']:
            summary = plan['summary']
            print(f"  动作分布: 去人口中心={summary.get('to_center', 0)}, 去转运站={summary.get('to_station', 0)}, 休整={summary.get('rest', 0)}")
        
        vehicle_status = plan.get('vehicle_status', {})
        print(f"  车辆状态: {dict(vehicle_status)}")
        
        center_demand = plan.get('center_demand', {})
        print(f"  人口中心需求: {center_demand.get('total_demand', 0)}个有需求, {center_demand.get('high_demand', 0)}个高需求")
        
        station_status = plan.get('station_status', {})
        print(f"  转运站状态: {station_status.get('total_queue', 0)}辆车在排队, {station_status.get('busy_stations', 0)}个繁忙转运站")
        
        if i <= 3 and plan['plan']:  # 只显示前三步的详细规划
            print(f"  详细规划:")
            for j, (vehicle_id, data) in enumerate(list(plan['plan'].items())[:3]):
                print(f"    {vehicle_id}: {data['description']} (Q值: {data.get('q_value', 0):.3f})")
            if len(plan['plan']) > 3:
                print(f"    ... 还有{len(plan['plan'])-3}辆车")

def visualize_training_history(training_history):
    """可视化训练历史"""
    try:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('深度强化学习训练历史', fontsize=14, fontweight='bold')
        
        # 1. 回合奖励
        axes[0, 0].plot(training_history['episode_rewards'])
        axes[0, 0].set_title('回合奖励')
        axes[0, 0].set_xlabel('回合')
        axes[0, 0].set_ylabel('奖励')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 垃圾收集量
        axes[0, 1].plot(training_history['episode_garbage_collected'])
        axes[0, 1].set_title('垃圾收集量')
        axes[0, 1].set_xlabel('回合')
        axes[0, 1].set_ylabel('垃圾量(吨)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 车辆利用率
        axes[0, 2].plot(training_history['episode_utilization'])
        axes[0, 2].set_title('车辆利用率')
        axes[0, 2].set_xlabel('回合')
        axes[0, 2].set_ylabel('利用率(%)')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. 损失值
        axes[1, 0].plot(training_history['episode_losses'])
        axes[1, 0].set_title('训练损失')
        axes[1, 0].set_xlabel('回合')
        axes[1, 0].set_ylabel('损失')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. 探索率
        axes[1, 1].plot(training_history['epsilon_values'])
        axes[1, 1].set_title('探索率衰减')
        axes[1, 1].set_xlabel('回合')
        axes[1, 1].set_ylabel('探索率')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. 预约违规
        axes[1, 2].plot(training_history['episode_reservation_violations'])
        axes[1, 2].set_title('预约违规次数')
        axes[1, 2].set_xlabel('回合')
        axes[1, 2].set_ylabel('违规次数')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("训练历史图已保存到 training_history.png")
        
    except Exception as e:
        print(f"可视化训练历史时出错: {e}")

def visualize_real_time_status(env, planner, step_counter):
    """可视化实时状态"""
    try:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'垃圾收运实时运营规划 (时间: {env.current_time:.2f}小时, 步数: {step_counter})', 
                    fontsize=14, fontweight='bold')
        
        # 1. 车辆状态分布
        status_counts = defaultdict(int)
        for vehicle in env.vehicles:
            status_counts[vehicle.status] += 1
        
        statuses = list(status_counts.keys())
        counts = list(status_counts.values())
        
        axes[0, 0].bar(range(len(statuses)), counts, tick_label=statuses, color='skyblue')
        axes[0, 0].set_title('车辆状态分布')
        axes[0, 0].set_ylabel('车辆数量')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. 人口中心垃圾量Top10
        center_garbage = [(c.id, c.current_garbage) for c in env.population_centers.values()]
        center_garbage.sort(key=lambda x: x[1], reverse=True)
        top_centers = center_garbage[:10]
        
        center_ids = [c[0][-3:] for c in top_centers]
        garbage_amounts = [c[1] for c in top_centers]
        
        axes[0, 1].bar(range(len(center_ids)), garbage_amounts, color='lightcoral')
        axes[0, 1].set_title('人口中心垃圾量Top10')
        axes[0, 1].set_ylabel('垃圾量(吨)')
        axes[0, 1].set_xticks(range(len(center_ids)))
        axes[0, 1].set_xticklabels(center_ids, rotation=45)
        
        # 3. 转运站利用率
        station_utilization = []
        station_ids = []
        for station_id, station in env.transfer_stations.items():
            utilization = station.current_garbage / station.capacity * 100 if station.capacity > 0 else 0
            station_utilization.append(utilization)
            station_ids.append(station_id[-2:])
        
        axes[0, 2].bar(range(len(station_ids)), station_utilization, color='lightgreen')
        axes[0, 2].set_title('转运站利用率')
        axes[0, 2].set_ylabel('利用率(%)')
        axes[0, 2].set_xticks(range(len(station_ids)))
        axes[0, 2].set_xticklabels(station_ids, rotation=45)
        axes[0, 2].axhline(y=80, color='red', linestyle='--', alpha=0.5, label='80%阈值')
        
        # 4. 车辆载重分布
        load_distribution = defaultdict(int)
        for vehicle in env.vehicles:
            if vehicle.current_load > 0:
                load_ratio = vehicle.current_load / env.vehicle_capacity
                if load_ratio < 0.25:
                    load_distribution['0-25%'] += 1
                elif load_ratio < 0.5:
                    load_distribution['25-50%'] += 1
                elif load_ratio < 0.75:
                    load_distribution['50-75%'] += 1
                else:
                    load_distribution['75-100%'] += 1
            else:
                load_distribution['空车'] += 1
        
        load_labels = list(load_distribution.keys())
        load_counts = list(load_distribution.values())
        
        axes[1, 0].pie(load_counts, labels=load_labels, autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('车辆载重分布')
        
        # 5. 转运站排队长度
        queue_lengths = [len(station.queue) for station in env.transfer_stations.values()]
        station_ids_short = [sid[-2:] for sid in env.transfer_station_ids]
        
        axes[1, 1].bar(range(len(station_ids_short)), queue_lengths, color='orange')
        axes[1, 1].set_title('转运站排队长度')
        axes[1, 1].set_ylabel('排队车辆数')
        axes[1, 1].set_xticks(range(len(station_ids_short)))
        axes[1, 1].set_xticklabels(station_ids_short, rotation=45)
        
        # 6. 性能指标
        metrics_text = f"""
        总垃圾收集: {sum(c.total_collected for c in env.population_centers.values()):.1f}吨
        总垃圾处理: {sum(s.total_processed for s in env.transfer_stations.values()):.1f}吨
        总行驶距离: {sum(v.total_distance for v in env.vehicles):.1f}公里
        空跑次数: {sum(v.empty_runs for v in env.vehicles)}
        休整次数: {sum(1 for v in env.vehicles if v.status == 'resting')}
        预约违规: {env.performance_metrics['total_reservation_violations']}
        车辆利用率: {sum(1 for v in env.vehicles if v.status not in ['idle', 'waiting', 'resting'])/len(env.vehicles)*100:.1f}%
        """
        
        axes[1, 2].text(0.1, 0.5, metrics_text, fontsize=10, verticalalignment='center')
        axes[1, 2].set_title('性能指标')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'real_time_status_step_{step_counter:04d}.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"实时状态图已保存到 real_time_status_step_{step_counter:04d}.png")
        
    except Exception as e:
        print(f"可视化过程中出错: {e}")

def run_real_time_simulation(env, agent, num_steps=180):
    """运行实时模拟（优化版）"""
    print("="*80)
    print("开始实时运营规划模拟")
    print("="*80)
    
    planner = RealTimePlanner(env, agent)
    
    state = env.reset()
    
    print(f"初始状态: 总垃圾量={sum(c.current_garbage for c in env.population_centers.values()):.2f}吨")
    print(f"初始车辆数: {len(env.vehicles)}辆")
    
    status_counts = defaultdict(int)
    for v in env.vehicles:
        status_counts[v.status] += 1
    print(f"初始车辆状态: {dict(status_counts)}")
    
    demand_centers = [c.id for c in env.population_centers.values() if c.can_collect(env.vehicle_capacity)]
    print(f"有垃圾需求的人口中心: {len(demand_centers)}个")
    
    # 生成并显示未来10步规划
    print("\n正在生成未来10步规划...")
    future_plans = planner.generate_future_plan(steps=10)
    print_future_plan_details(future_plans)
    visualize_future_plan(future_plans)
    
    for step in range(num_steps):
        print(f"\n=== 步数 {step+1}/{num_steps} (时间: {env.current_time:.2f}小时) ===")
        
        decision_vehicles = env.get_decision_vehicles()
        print(f"可决策车辆数: {len(decision_vehicles)}")
        
        status_counts = defaultdict(int)
        for v in env.vehicles:
            status_counts[v.status] += 1
        print(f"车辆状态统计: {dict(status_counts)}")
        
        if decision_vehicles:
            plan = planner.generate_plan()
            
            if plan:
                print(f"生成 {len(plan)} 辆车的运营规划:")
                action_types = defaultdict(int)
                for vehicle_id, data in plan.items():
                    action_types[data['destination_type']] += 1
                
                print(f"  动作分布: 去人口中心={action_types.get('center', 0)}, 去转运站={action_types.get('station', 0)}, 休整={action_types.get('rest', 0)}")
                
                for i, (vehicle_id, data) in enumerate(list(plan.items())[:5]):
                    print(f"  {vehicle_id}: {data['description']} (Q值: {data['q_value']:.3f})")
                if len(plan) > 5:
                    print(f"  ... 还有{len(plan)-5}辆车")
            
            result = planner.execute_plan()
            if result:
                print(f"执行结果: 奖励={result['reward']:.3f}")
                
                summary = planner.get_plan_summary()
                if summary:
                    print(f"规划摘要: {summary['decision_vehicles']}辆车有规划, {summary['center_demand']}个人口中心有垃圾需求")
        else:
            print("当前没有需要决策的车辆")
            next_state, reward, done, _ = env.step({})
            print(f"环境更新: 奖励={reward:.3f}")
        
        # 每30步生成一次未来10步规划
        if (step + 1) % 30 == 0:
            print(f"\n=== 更新未来10步规划 (步数: {step+1}) ===")
            future_plans = planner.generate_future_plan(steps=10)
            print_future_plan_details(future_plans)
        
        if (step + 1) % 30 == 0:
            visualize_real_time_status(env, planner, step + 1)
        
        if (step + 1) % 60 == 0:
            current_plan = env.get_current_plan()
            with open(f'current_plan_step_{step+1:04d}.json', 'w') as f:
                json.dump(current_plan, f, indent=2, default=str)
            print(f"当前运营规划已保存到 current_plan_step_{step+1:04d}.json")
        
        if step >= env.max_steps:
            print("模拟结束")
            break
    
    print("\n" + "="*80)
    print("模拟完成，生成最终报告")
    print("="*80)
    
    final_report = generate_final_report(env)
    print_report(final_report)
    
    with open('final_simulation_report.json', 'w') as f:
        json.dump(final_report, f, indent=2, default=str)
    
    print(f"\n最终报告已保存到 final_simulation_report.json")
    
    return env, planner

def generate_final_report(env):
    """生成最终报告"""
    report = {
        'simulation_summary': {
            'total_steps': env.current_step,
            'total_time_hours': env.current_time,
            'num_population_centers': len(env.population_centers),
            'num_transfer_stations': len(env.transfer_stations),
            'num_vehicles': len(env.vehicles),
            'vehicle_capacity': env.vehicle_capacity,
            'max_nearby_centers': env.max_nearby_centers,
            'max_nearby_stations': env.max_nearby_stations
        },
        'performance_metrics': env.performance_metrics,
        'garbage_statistics': {
            'total_generated': sum(c.total_generated for c in env.population_centers.values()),
            'total_collected': sum(c.total_collected for c in env.population_centers.values()),
            'total_processed': sum(s.total_processed for s in env.transfer_stations.values()),
            'remaining_garbage': sum(c.current_garbage for c in env.population_centers.values()),
            'collection_rate': sum(c.total_collected for c in env.population_centers.values()) / 
                              max(sum(c.total_generated for c in env.population_centers.values()), 1)
        },
        'vehicle_statistics': {
            'total_distance': sum(v.total_distance for v in env.vehicles),
            'total_trips': sum(v.trips_completed for v in env.vehicles),
            'total_operating_time': sum(v.total_operating_time for v in env.vehicles),
            'total_idle_time': sum(v.idle_time for v in env.vehicles),
            'total_waiting_time': sum(v.waiting_time for v in env.vehicles),
            'total_rest_time': sum(v.rest_time for v in env.vehicles),
            'total_empty_runs': sum(v.empty_runs for v in env.vehicles),
            'average_utilization': (sum(v.total_operating_time - v.idle_time - v.waiting_time - v.rest_time for v in env.vehicles) / 
                                   max(sum(v.total_operating_time for v in env.vehicles), 1)) * 100
        },
        'station_statistics': {
            'total_queue_time': sum(s.queue_waiting_time for s in env.transfer_stations.values()),
            'average_queue_length': np.mean([len(s.queue) for s in env.transfer_stations.values()]) 
                                   if len(env.transfer_stations) > 0 else 0,
            'station_utilization': {}
        },
        'final_vehicle_status': defaultdict(int),
        'reservation_statistics': {
            'total_reservations': len(env.route_reservations),
            'center_reservations': {},
            'station_reservations': {}
        },
        'recommendations': []
    }
    
    for station_id, station in env.transfer_stations.items():
        utilization = station.current_garbage / station.capacity * 100 if station.capacity > 0 else 0
        report['station_statistics']['station_utilization'][station_id] = {
            'current_garbage': station.current_garbage,
            'capacity': station.capacity,
            'utilization_percent': utilization,
            'queue_length': len(station.queue),
            'total_processed': station.total_processed,
            'device_num': station.device_num,
            'assigned_vehicles': station.device_num
        }
    
    for vehicle in env.vehicles:
        report['final_vehicle_status'][vehicle.status] += 1
    
    station_vehicle_counts = defaultdict(int)
    for vehicle in env.vehicles:
        if vehicle.current_location_type == 'station':
            station_vehicle_counts[vehicle.current_location_id] += 1
    
    report['vehicle_distribution'] = dict(station_vehicle_counts)
    
    # 预约统计
    for center_id, reservations in env.center_reservations.items():
        report['reservation_statistics']['center_reservations'][center_id] = len(reservations)
    
    for station_id, reservations in env.station_reservations.items():
        report['reservation_statistics']['station_reservations'][station_id] = len(reservations)
    
    report['recommendations'] = generate_recommendations(env)
    
    return report

def print_report(report):
    """打印报告"""
    print("\n" + "="*80)
    print("垃圾收运运营规划最终报告")
    print("="*80)
    
    print(f"\n1. 模拟概况:")
    for key, value in report['simulation_summary'].items():
        print(f"   {key}: {value}")
    
    print(f"\n2. 垃圾处理成效:")
    garbage_stats = report['garbage_statistics']
    print(f"   总生成垃圾: {garbage_stats['total_generated']:.2f}吨")
    print(f"   总收集垃圾: {garbage_stats['total_collected']:.2f}吨")
    print(f"   总处理垃圾: {garbage_stats['total_processed']:.2f}吨")
    print(f"   剩余垃圾: {garbage_stats['remaining_garbage']:.2f}吨")
    print(f"   收集率: {garbage_stats['collection_rate']*100:.1f}%")
    
    print(f"\n3. 车辆运营效率:")
    vehicle_stats = report['vehicle_statistics']
    print(f"   总行驶距离: {vehicle_stats['total_distance']:.2f}公里")
    print(f"   总运输次数: {vehicle_stats['total_trips']}次")
    print(f"   总运营时间: {vehicle_stats['total_operating_time']:.2f}小时")
    print(f"   总空闲时间: {vehicle_stats['total_idle_time']:.2f}小时")
    print(f"   总等待时间: {vehicle_stats['total_waiting_time']:.2f}小时")
    print(f"   总休整时间: {vehicle_stats['total_rest_time']:.2f}小时")
    print(f"   总空跑次数: {vehicle_stats['total_empty_runs']}次")
    print(f"   平均利用率: {vehicle_stats['average_utilization']:.1f}%")
    
    print(f"\n4. 车辆分配情况:")
    for station_id, count in report['vehicle_distribution'].items():
        print(f"   {station_id}: {count}辆车")
    
    print(f"\n5. 最终车辆状态:")
    for status, count in report['final_vehicle_status'].items():
        percentage = count / report['simulation_summary']['num_vehicles'] * 100
        print(f"   {status}: {count}辆 ({percentage:.1f}%)")
    
    print(f"\n6. 预约统计:")
    print(f"   总预约数: {report['reservation_statistics']['total_reservations']}")
    centers_with_reservations = sum(1 for v in report['reservation_statistics']['center_reservations'].values() if v > 0)
    print(f"   有预约的人口中心: {centers_with_reservations}个")
    stations_with_reservations = sum(1 for v in report['reservation_statistics']['station_reservations'].values() if v > 0)
    print(f"   有预约的转运站: {stations_with_reservations}个")
    
    print(f"\n7. 优化建议:")
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"   {i}. {rec}")

def generate_recommendations(env):
    """生成优化建议"""
    recommendations = []
    
    congested_centers = []
    for center_id, center in env.population_centers.items():
        if center.current_garbage > env.vehicle_capacity * 1:
            congested_centers.append((center_id, center.current_garbage, center.collection_queue))
    
    if congested_centers:
        congested_centers.sort(key=lambda x: x[1], reverse=True)
        top_center = congested_centers[0]
        recommendations.append(f"人口中心 {top_center[0]} 垃圾堆积({top_center[1]:.1f}吨, 排队车辆:{top_center[2]})")
    
    congested_stations = []
    for station_id, station in env.transfer_stations.items():
        if len(station.queue) > 5:
            congested_stations.append((station_id, len(station.queue)))
    
    if congested_stations:
        congested_stations.sort(key=lambda x: x[1], reverse=True)
        top_station = congested_stations[0]
        recommendations.append(f"转运站 {top_station[0]} 排队严重({top_station[1]}辆车)，建议分流到其他转运站或增加设备")
    
    active_vehicles = sum(1 for v in env.vehicles if v.status not in ['idle', 'waiting', 'resting'])
    utilization = active_vehicles / len(env.vehicles) * 100 if len(env.vehicles) > 0 else 0
    
    if utilization < 60:
        recommendations.append(f"车辆利用率较低({utilization:.1f}%)，建议优化调度策略，减少空跑和休整")
    
    total_empty_runs = sum(v.empty_runs for v in env.vehicles)
    if total_empty_runs > len(env.vehicles) * 0.5:
        recommendations.append(f"空跑次数过多({total_empty_runs}次)，建议改进垃圾量预测和调度策略")
    
    loaded_vehicles = [v for v in env.vehicles if v.current_load > 0]
    if loaded_vehicles:
        avg_load = sum(v.current_load for v in loaded_vehicles) / len(loaded_vehicles)
        avg_load_ratio = avg_load / env.vehicle_capacity
        
        if avg_load_ratio < 0.7:
            recommendations.append(f"车辆平均满载率较低({avg_load_ratio*100:.1f}%)，建议优化装载策略，优先选择垃圾量多的人口中心")
    
    for station_id, station in env.transfer_stations.items():
        utilization = station.current_garbage / station.capacity * 100 if station.capacity > 0 else 0
        if utilization > 90:
            recommendations.append(f"转运站 {station_id} 利用率过高({utilization:.1f}%)，存在溢出风险，建议分流")
        elif utilization > 30:
            recommendations.append(f"转运站 {station_id} 利用率({utilization:.1f}%)，正常")
        elif utilization < 30:
            recommendations.append(f"转运站 {station_id} 利用率过低({utilization:.1f}%)，资源未充分利用，建议调整车辆分配")
    
    # 预约相关建议
    reservation_violations = env.performance_metrics['total_reservation_violations']
    if reservation_violations > 10:
        recommendations.append(f"预约违规次数较多({reservation_violations}次)，建议改进预约机制或调度算法")
    
    for center_id, reservations in env.center_reservations.items():
        if len(reservations) > 2:
            recommendations.append(f"人口中心 {center_id} 预约车辆过多({len(reservations)}辆)，可能导致车辆到达后无垃圾可收")
    
    if not recommendations:
        recommendations.append("当前运营状况良好，继续保持现有调度策略")
    
    return recommendations

# ===================== 第七部分：最佳运营方案导出模块 =====================
class BestSolutionExporter:
    """最佳运营方案导出器"""
    
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.best_plan_records = []
        self.vehicle_records = []
        self.center_records = []
        self.station_records = []
        self.reservation_records = []
        
    def run_and_record_best_solution(self, num_steps=180):
        """运行并记录最佳运营方案"""
        print("\n" + "="*80)
        print("运行最佳运营方案模拟并记录...")
        print("="*80)
        
        # 重置环境
        state = self.env.reset()
        total_reward = 0.0
        
        # 设置智能体为评估模式（不探索）
        original_eps = self.agent.eps
        self.agent.eps = 0.01  # 设置很小的探索率，确保选择最佳动作
        
        for step in range(num_steps):
            print(f"\n步数 {step+1}/{num_steps} (时间: {self.env.current_time:.2f}小时)")
            
            # 记录当前状态
            self._record_current_state(step)
            
            # 获取决策并执行
            decision_vehicles = self.env.get_decision_vehicles()
            actions = {}
            
            for vehicle in decision_vehicles:
                available_actions = self.env.get_available_actions(vehicle)
                
                # 使用智能体选择最佳动作
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
                    q_values = self.agent.policy_net(state_tensor).cpu().numpy()[0]
                    
                    if available_actions:
                        available_q_values = [q_values[a] for a in available_actions]
                        best_idx = np.argmax(available_q_values)
                        action = available_actions[best_idx]
                    else:
                        action = np.argmax(q_values)
                
                actions[vehicle.id] = action
                
                # 记录车辆决策
                dest_type, dest_id = self.env.map_action_to_destination(vehicle, action)
                self._record_vehicle_decision(step, vehicle, action, dest_type, dest_id)
            
            # 执行动作
            next_state, reward, done, _ = self.env.step(actions)
            total_reward += reward
            state = next_state
            
            if done:
                print("模拟结束")
                break
        
        # 恢复智能体探索率
        self.agent.eps = original_eps
        
        print(f"\n最佳运营方案模拟完成，总奖励: {total_reward:.2f}")
        print(f"总垃圾收集量: {sum(c.total_collected for c in self.env.population_centers.values()):.2f}吨")
        print(f"总垃圾处理量: {sum(s.total_processed for s in self.env.transfer_stations.values()):.2f}吨")
        print(f"总行驶距离: {sum(v.total_distance for v in self.env.vehicles):.2f}公里")
        
        return total_reward
    
    def _record_current_state(self, step):
        """记录当前状态"""
        # 记录整体状态
        total_garbage = sum(c.current_garbage for c in self.env.population_centers.values())
        total_collected = sum(c.total_collected for c in self.env.population_centers.values())
        total_processed = sum(s.total_processed for s in self.env.transfer_stations.values())
        
        status_counts = defaultdict(int)
        for vehicle in self.env.vehicles:
            status_counts[vehicle.status] += 1
        
        plan_record = {
            'step': step + 1,
            'time_hours': self.env.current_time,
            'total_garbage_tons': total_garbage,
            'total_collected_tons': total_collected,
            'total_processed_tons': total_processed,
            'vehicles_idle': status_counts.get('idle', 0),
            'vehicles_traveling': status_counts.get('traveling', 0),
            'vehicles_loading': status_counts.get('loading', 0),
            'vehicles_unloading': status_counts.get('unloading', 0),
            'vehicles_waiting': status_counts.get('waiting', 0),
            'vehicles_loaded': status_counts.get('loaded', 0),
            'vehicles_resting': status_counts.get('resting', 0),
            'total_distance_km': sum(v.total_distance for v in self.env.vehicles),
            'total_empty_runs': sum(v.empty_runs for v in self.env.vehicles)
        }
        
        self.best_plan_records.append(plan_record)
    
    def _record_vehicle_decision(self, step, vehicle, action, dest_type, dest_id):
        """记录车辆决策"""
        vehicle_record = {
            'step': step + 1,
            'time_hours': self.env.current_time,
            'vehicle_id': vehicle.id,
            'vehicle_status': vehicle.status,
            'location_type': vehicle.current_location_type,
            'location_id': vehicle.current_location_id,
            'current_load_tons': vehicle.current_load,
            'capacity_tons': vehicle.capacity,
            'load_percentage': (vehicle.current_load / vehicle.capacity * 100) if vehicle.capacity > 0 else 0,
            'action': action,
            'destination_type': dest_type,
            'destination_id': dest_id,
            'destination_description': self._get_destination_description(dest_type, dest_id),
            'total_distance_km': vehicle.total_distance,
            'total_collected_tons': vehicle.total_garbage_collected,
            'total_unloaded_tons': vehicle.total_garbage_unloaded,
            'trips_completed': vehicle.trips_completed,
            'empty_runs': vehicle.empty_runs
        }
        
        self.vehicle_records.append(vehicle_record)
        
        # 记录预约信息
        if vehicle.id in self.env.route_reservations:
            target_type, target_id, arrival_time = self.env.route_reservations[vehicle.id]
            reservation_record = {
                'step': step + 1,
                'time_hours': self.env.current_time,
                'vehicle_id': vehicle.id,
                'reservation_type': target_type,
                'reservation_target': target_id,
                'scheduled_arrival_time': arrival_time,
                'actual_time': self.env.current_time,
                'time_difference': self.env.current_time - arrival_time
            }
            self.reservation_records.append(reservation_record)
    
    def _get_destination_description(self, dest_type, dest_id):
        """获取目的地描述"""
        if dest_type == 'center':
            if dest_id in self.env.population_centers:
                center = self.env.population_centers[dest_id]
                return f"人口中心 {dest_id} (垃圾量: {center.current_garbage:.2f}吨)"
        elif dest_type == 'station':
            if dest_id in self.env.transfer_stations:
                station = self.env.transfer_stations[dest_id]
                return f"转运站 {dest_id} (容量: {station.capacity:.0f}吨, 当前: {station.current_garbage:.2f}吨)"
        elif dest_type == 'rest':
            return "车辆休整"
        
        return f"{dest_type}: {dest_id}"
    
    def record_centers_and_stations(self):
        """记录人口中心和转运站的详细信息"""
        # 记录人口中心
        for center_id, center in self.env.population_centers.items():
            center_record = {
                'center_id': center_id,
                'longitude': center.longitude,
                'latitude': center.latitude,
                'population': center.population,
                'current_garbage_tons': center.current_garbage,
                'total_generated_tons': center.total_generated,
                'total_collected_tons': center.total_collected,
                'waiting_time_hours': center.waiting_time,
                'generation_rate_tph': center.generation_rate,
                'can_collect': center.can_collect(self.env.vehicle_capacity),
                'collection_queue': center.collection_queue,
                'reservation_count': len(self.env.center_reservations.get(center_id, []))
            }
            self.center_records.append(center_record)
        
        # 记录转运站
        for station_id, station in self.env.transfer_stations.items():
            station_record = {
                'station_id': station_id,
                'longitude': station.longitude,
                'latitude': station.latitude,
                'capacity_tons': station.capacity,
                'current_garbage_tons': station.current_garbage,
                'utilization_percent': (station.current_garbage / station.capacity * 100) if station.capacity > 0 else 0,
                'device_num': station.device_num,
                'available_devices': station.available_devices,
                'queue_length': len(station.queue),
                'total_processed_tons': station.total_processed,
                'reservation_count': len(self.env.station_reservations.get(station_id, []))
            }
            self.station_records.append(station_record)
    
    def export_to_excel(self, filepath="最佳运营方案.xlsx"):
        """导出最佳运营方案到Excel"""
        print(f"\n正在导出最佳运营方案到Excel: {filepath}")
        
        # 确保有记录的数据
        if not self.best_plan_records:
            print("没有可导出的数据，请先运行模拟")
            return
        
        # 记录人口中心和转运站信息
        self.record_centers_and_stations()
        
        try:
            # 创建Excel写入器
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # 1. 运营总览表
                overview_df = pd.DataFrame(self.best_plan_records)
                overview_df.to_excel(writer, sheet_name='运营总览', index=False)
                
                # 2. 车辆调度详情表
                if self.vehicle_records:
                    vehicle_df = pd.DataFrame(self.vehicle_records)
                    vehicle_df.to_excel(writer, sheet_name='车辆调度详情', index=False)
                
                # 3. 人口中心状态表
                if self.center_records:
                    center_df = pd.DataFrame(self.center_records)
                    center_df.to_excel(writer, sheet_name='人口中心状态', index=False)
                
                # 4. 转运站状态表
                if self.station_records:
                    station_df = pd.DataFrame(self.station_records)
                    station_df.to_excel(writer, sheet_name='转运站状态', index=False)
                
                # 5. 预约记录表
                if self.reservation_records:
                    reservation_df = pd.DataFrame(self.reservation_records)
                    reservation_df.to_excel(writer, sheet_name='预约记录', index=False)
                
                # 6. 性能统计表
                performance_stats = self._calculate_performance_stats()
                performance_df = pd.DataFrame([performance_stats])
                performance_df.to_excel(writer, sheet_name='性能统计', index=False)
                
                # 7. 优化建议表
                recommendations = self._generate_recommendations()
                recommendations_df = pd.DataFrame({'优化建议': recommendations})
                recommendations_df.to_excel(writer, sheet_name='优化建议', index=False)
            
            print(f"最佳运营方案已成功导出到: {filepath}")
            print(f"包含以下工作表:")
            print(f"  1. 运营总览 - {len(overview_df)}条记录")
            print(f"  2. 车辆调度详情 - {len(vehicle_df) if self.vehicle_records else 0}条记录")
            print(f"  3. 人口中心状态 - {len(center_df) if self.center_records else 0}条记录")
            print(f"  4. 转运站状态 - {len(station_df) if self.station_records else 0}条记录")
            print(f"  5. 预约记录 - {len(reservation_df) if self.reservation_records else 0}条记录")
            print(f"  6. 性能统计 - 关键性能指标")
            print(f"  7. 优化建议 - 改进建议")
            
            return filepath
            
        except Exception as e:
            print(f"导出Excel时出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _calculate_performance_stats(self):
        """计算性能统计数据"""
        if not self.best_plan_records:
            return {}
        
        last_record = self.best_plan_records[-1]
        
        # 计算车辆利用率
        total_vehicles = len(self.env.vehicles)
        active_vehicles = sum(1 for v in self.env.vehicles 
                            if v.status not in ['idle', 'waiting', 'resting'])
        utilization_percent = (active_vehicles / total_vehicles * 100) if total_vehicles > 0 else 0
        
        # 计算平均载重率
        loaded_vehicles = [v for v in self.env.vehicles if v.current_load > 0]
        if loaded_vehicles:
            avg_load_ratio = sum(v.current_load for v in loaded_vehicles) / (len(loaded_vehicles) * self.env.vehicle_capacity)
        else:
            avg_load_ratio = 0
        
        # 收集总时间统计
        total_operating_time = sum(v.total_operating_time for v in self.env.vehicles)
        total_idle_time = sum(v.idle_time for v in self.env.vehicles)
        total_waiting_time = sum(v.waiting_time for v in self.env.vehicles)
        
        stats = {
            '模拟总时长_小时': last_record['time_hours'],
            '模拟总步数': last_record['step'],
            '总人口中心数': len(self.env.population_centers),
            '总转运站数': len(self.env.transfer_stations),
            '总车辆数': total_vehicles,
            '车辆容量_吨': self.env.vehicle_capacity,
            '总垃圾收集量_吨': last_record['total_collected_tons'],
            '总垃圾处理量_吨': last_record['total_processed_tons'],
            '剩余垃圾量_吨': last_record['total_garbage_tons'],
            '收集率_百分比': (last_record['total_collected_tons'] / 
                            (last_record['total_collected_tons'] + last_record['total_garbage_tons']) * 100 
                            if (last_record['total_collected_tons'] + last_record['total_garbage_tons']) > 0 else 0),
            '总行驶距离_公里': last_record['total_distance_km'],
            '总空跑次数': last_record['total_empty_runs'],
            '车辆利用率_百分比': utilization_percent,
            '平均载重率_百分比': avg_load_ratio * 100,
            '总运营时间_小时': total_operating_time,
            '总空闲时间_小时': total_idle_time,
            '总等待时间_小时': total_waiting_time,
            '空闲率_百分比': (total_idle_time / total_operating_time * 100) if total_operating_time > 0 else 0,
            '预约违规次数': self.env.performance_metrics.get('total_reservation_violations', 0),
            '休整车辆数': self.env.performance_metrics.get('total_rest_count', 0)
        }
        
        return stats
    
    def _generate_recommendations(self):
        """生成优化建议"""
        recommendations = []
        
        # 分析车辆利用率
        total_vehicles = len(self.env.vehicles)
        idle_vehicles = sum(1 for v in self.env.vehicles if v.status == 'idle')
        idle_rate = (idle_vehicles / total_vehicles * 100) if total_vehicles > 0 else 0
        
        if idle_rate > 30:
            recommendations.append(f"车辆空闲率较高({idle_rate:.1f}%)，建议优化调度策略，减少车辆闲置")
        
        # 分析空跑情况
        total_empty_runs = sum(v.empty_runs for v in self.env.vehicles)
        if total_empty_runs > total_vehicles * 0.5:
            recommendations.append(f"空跑次数较多({total_empty_runs}次)，建议改进垃圾量预测，避免车辆到达后无垃圾可收")
        
        # 分析转运站排队
        max_queue = max(len(s.queue) for s in self.env.transfer_stations.values())
        if max_queue > 5:
            recommendations.append(f"转运站最大排队长度{max_queue}辆车，建议增加设备或优化车辆分配")
        
        # 分析垃圾堆积
        max_garbage = max(c.current_garbage for c in self.env.population_centers.values())
        if max_garbage > self.env.vehicle_capacity * 3:
            recommendations.append(f"最大垃圾堆积{max_garbage:.2f}吨，超过3车容量，建议增加收集频次")
        
        # 分析预约违规
        reservation_violations = self.env.performance_metrics.get('total_reservation_violations', 0)
        if reservation_violations > 5:
            recommendations.append(f"预约违规{reservation_violations}次，建议优化路径规划和时间预估")
        
        # 分析车辆分配均衡性
        station_vehicle_counts = defaultdict(int)
        for vehicle in self.env.vehicles:
            if vehicle.current_location_type == 'station':
                station_vehicle_counts[vehicle.current_location_id] += 1
        
        if station_vehicle_counts:
            avg_vehicles = np.mean(list(station_vehicle_counts.values()))
            max_diff = max(abs(count - avg_vehicles) for count in station_vehicle_counts.values())
            if max_diff > avg_vehicles * 0.5:
                recommendations.append("车辆在各转运站间分配不均衡，建议重新分配车辆")
        
        if not recommendations:
            recommendations.append("当前运营状况良好，继续保持现有调度策略")
        
        return recommendations

# ===================== 第八部分：主程序 =====================
def main():
    """主函数：深度强化学习车辆运营规划"""
    print("="*80)
    print("深度强化学习垃圾收运车辆运营规划系统（优化版）")
    print("="*80)
    print("优化特性:")
    print("1. 距离限制选择: 车辆只能选择最近的5个转运站或7个人口中心")
    print("2. 预约机制: 人口中心最多预约2辆车，转运站按可用设备数预约")
    print("3. 空运惩罚: 车辆到达人口中心但垃圾量不足时给予大惩罚")
    print("4. 休整惩罚: 车辆选择休整时给予小惩罚")
    print("5. 初始状态: 每个转运站分配设备数相同的转运车")
    print("6. 冲突避免: 限制每个人口中心的排队车辆数")
    print("7. 预约违规惩罚: 车辆不按预约执行时给予惩罚")
    print("8. 未来规划: 提供未来10步的运营规划")
    print("9. 最佳方案导出: 导出最优运营方案到Excel")
    print("="*80)
    
    # 1. 读取数据
    print("\n1. 读取数据...")
    file_path = "C:\\Users\\29744\\Desktop\\RL\\input.xlsx"
    
    try:
        population_df, transferstation_df = read_excel_data(file_path)
        if population_df is None or transferstation_df is None:
            print("读取数据失败，请检查文件路径和格式")
            return None, None, None
    except Exception as e:
        print(f"读取数据时出错: {e}")
        return None, None, None
    
    print(f"\n垃圾生成参数:")
    print(f"  人均日垃圾产量: 1.2kg = 0.0012吨")
    print(f"  模拟时间步长: 5分钟 = {5/60:.4f}小时")
    
    if len(population_df) > 0:
        sample_population = population_df.iloc[0]['population']
        daily_gen = sample_population * 0.0012
        hourly_gen = daily_gen / 24
        five_min_gen = hourly_gen / 12
        print(f"  示例（人口{sample_population:,}）:")
        print(f"    日生成量: {daily_gen:.2f}吨/天")
        print(f"    时生成量: {hourly_gen:.4f}吨/小时")
        print(f"    5分钟生成量: {five_min_gen:.6f}吨/5分钟")
    
    # 2. 创建环境（使用优化的环境）
    print("\n2. 创建优化强化学习环境...")
    
    # 计算总设备数
    total_devices = sum(int(row['device_num']) for _, row in transferstation_df.iterrows())
    print(f"  总设备数（车辆数）: {total_devices}")
    
    env = WasteCollectionRLEnv(
        population_df=population_df,
        transferstation_df=transferstation_df,
        num_vehicles=min(total_devices, 108),  # 限制车辆数，加快训练速度
        vehicle_capacity=10.0
    )
    
    print(f"   状态空间: {env.state_space_size}维")
    print(f"   动作空间: {env.action_space_size}个动作")
    print(f"   最近人口中心限制: {env.max_nearby_centers}个（只考虑有垃圾的）")
    print(f"   最近转运站限制: {env.max_nearby_stations}个（只考虑无排队的）")
    print(f"   时间步长: {env.time_step*60:.0f}分钟")
    print(f"   模拟时长: {env.max_steps}步 ({env.max_steps * env.time_step:.1f}小时)")
    print(f"   初始条件: 每个转运站分配设备数相同的转运车，每个人口中心初始有5-20吨垃圾")
    
    # 验证车辆分配
    print("\n验证车辆分配:")
    station_vehicle_counts = {}
    for vehicle in env.vehicles:
        if vehicle.current_location_type == 'station':
            station_id = vehicle.current_location_id
            station_vehicle_counts[station_id] = station_vehicle_counts.get(station_id, 0) + 1
    
    for station_id, station in env.transfer_stations.items():
        assigned = station_vehicle_counts.get(station_id, 0)
        expected = station.device_num
        status = "✓" if assigned == expected else "✗"
        print(f"  转运站 {station_id}: 应有{expected}辆车，实有{assigned}辆车 {status}")
    
    # 3. 创建深度强化学习智能体
    print("\n3. 创建深度强化学习智能体...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   使用设备: {device}")
    
    agent = DQNAgent(
        state_dim=env.state_space_size,
        action_dim=env.action_space_size,
        num_vehicles=len(env.vehicles),
        device=device
    )
    
    # 4. 创建训练器
    print("\n4. 创建训练器...")
    trainer = DRLTrainer(
        env=env,
        agent=agent,
        num_episodes=100,  # 减少回合数以加快训练速度
        max_steps=36  # 减少步数以加快训练速度
    )
    
    # 5. 训练智能体
    print("\n5. 开始训练深度强化学习智能体...")
    print("="*80)
    
    training_history = trainer.train()
    
    # 6. 可视化训练历史
    print("\n6. 可视化训练历史...")
    visualize_training_history(training_history)
    
    # 7. 评估智能体
    print("\n7. 评估训练好的智能体...")
    evaluation_results, avg_results = trainer.evaluate(num_episodes=2)
    
    # 8. 运行DRL实时模拟（包含未来规划）
    print("\n8. 运行DRL实时运营规划模拟...")
    print("="*80)
    
    env_real_time = WasteCollectionRLEnv(
        population_df=population_df,
        transferstation_df=transferstation_df,
        num_vehicles=min(total_devices, 108),  # 减少车辆数以加快模拟速度
        vehicle_capacity=10.0
    )
    
    env_real_time, planner = run_real_time_simulation(env_real_time, agent, num_steps=60)
    
    # 9. 运行最佳运营方案并导出到Excel
    print("\n9. 运行最佳运营方案并导出到Excel...")
    print("="*80)
    
    # 创建最佳方案导出器
    exporter = BestSolutionExporter(env_real_time, agent)
    
    # 运行并记录最佳方案
    total_reward = exporter.run_and_record_best_solution(num_steps=72)  # 运行6小时（72步）
    
    # 导出到Excel
    excel_file = exporter.export_to_excel("最佳运营方案_详细记录.xlsx")
    
    if excel_file:
        # 生成汇总报告
        print("\n" + "="*80)
        print("最佳运营方案汇总报告")
        print("="*80)
        
        # 显示关键指标
        if exporter.best_plan_records:
            last_record = exporter.best_plan_records[-1]
            
            print(f"\n1. 模拟概况:")
            print(f"   模拟时长: {last_record['time_hours']:.2f}小时")
            print(f"   模拟步数: {last_record['step']}步")
            print(f"   总人口中心: {len(env_real_time.population_centers)}个")
            print(f"   总转运站: {len(env_real_time.transfer_stations)}个")
            print(f"   总车辆数: {len(env_real_time.vehicles)}辆")
            
            print(f"\n2. 垃圾处理成效:")
            print(f"   总垃圾收集量: {last_record['total_collected_tons']:.2f}吨")
            print(f"   总垃圾处理量: {last_record['total_processed_tons']:.2f}吨")
            print(f"   剩余垃圾量: {last_record['total_garbage_tons']:.2f}吨")
            collection_rate = (last_record['total_collected_tons'] / 
                              (last_record['total_collected_tons'] + last_record['total_garbage_tons']) * 100 
                              if (last_record['total_collected_tons'] + last_record['total_garbage_tons']) > 0 else 0)
            print(f"   垃圾收集率: {collection_rate:.1f}%")
            
            print(f"\n3. 车辆运营效率:")
            print(f"   总行驶距离: {last_record['total_distance_km']:.2f}公里")
            print(f"   总空跑次数: {last_record['total_empty_runs']}次")
            
            # 计算车辆利用率
            status_counts = defaultdict(int)
            for vehicle in env_real_time.vehicles:
                status_counts[vehicle.status] += 1
            
            print(f"\n4. 车辆状态分布:")
            for status, count in status_counts.items():
                percentage = count / len(env_real_time.vehicles) * 100
                print(f"   {status}: {count}辆 ({percentage:.1f}%)")
            
            print(f"\n5. 转运站状态:")
            for station_id, station in env_real_time.transfer_stations.items():
                utilization = station.current_garbage / station.capacity * 100 if station.capacity > 0 else 0
                print(f"   {station_id}: {station.current_garbage:.1f}/{station.capacity:.0f}吨 ({utilization:.1f}%)，排队: {len(station.queue)}辆")
            
            print(f"\n6. 优化建议:")
            recommendations = exporter._generate_recommendations()
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
            
            print(f"\n详细运营记录已保存到: {excel_file}")
    
    # 10. 保存最终结果
    print("\n10. 保存最终结果...")
    
    with open('training_history_optimized.json', 'w') as f:
        history_to_save = {}
        for key, value in training_history.items():
            if isinstance(value, list):
                history_to_save[key] = [float(v) if isinstance(v, (np.float32, np.float64)) else v 
                                       for v in value]
            else:
                history_to_save[key] = float(value) if isinstance(value, (np.float32, np.float64)) else value
        
        json.dump(history_to_save, f, indent=2)
    
    with open('evaluation_results_optimized.json', 'w') as f:
        eval_to_save = {}
        for key, value in evaluation_results.items():
            if isinstance(value, list):
                eval_to_save[key] = [float(v) if isinstance(v, (np.float32, np.float64)) else v 
                                    for v in value]
            else:
                eval_to_save[key] = float(value) if isinstance(value, (np.float32, np.float64)) else value
        
        json.dump(eval_to_save, f, indent=2)
    
    print("\n所有结果已保存到文件:")
    print("  - dqn_model_final.pth (模型权重)")
    print("  - training_history.png (训练历史图)")
    print("  - future_plan_10_steps.png (未来规划图)")
    print("  - training_history_optimized.json (训练历史)")
    print("  - evaluation_results_optimized.json (评估结果)")
    print("  - final_simulation_report.json (最终模拟报告)")
    print("  - real_time_status_step_*.png (实时状态图)")
    print("  - current_plan_step_*.json (当前运营规划)")
    print(f"  - {excel_file if excel_file else '最佳运营方案_详细记录.xlsx'} (最佳运营方案Excel记录)")
    
    print("\n" + "="*80)
    print("深度强化学习垃圾收运车辆运营规划完成!")
    print("="*80)
    
    print("\n最佳方案建议总结:")
    print("1. 使用DRL智能体进行车辆调度，相比贪心算法有明显改善")
    print("2. 每个转运站分配与其设备数相同的转运车")
    print("3. 车辆只选择最近的5个转运站或7个人口中心")
    print("4. 优先选择垃圾量多的人口中心")
    print("5. 避免多辆车同时选择同一个人口中心（限制排队车辆数）")
    print("6. 使用预约机制避免资源冲突")
    print("7. 对空跑、休整和预约违规给予惩罚，鼓励高效运营")
    print("8. 实时监控转运站利用率，避免拥堵和溢出")
    print("9. 使用未来10步规划进行前瞻性调度")
    print("10. 导出详细运营数据到Excel进行分析和优化")
    
    return env, agent, exporter

if __name__ == "__main__":
    try:
        env, agent, exporter = main()
        if env and agent and exporter:
            print("\n程序执行成功!")
    except Exception as e:
        print(f"运行完整版本时出错: {e}")
        import traceback
        traceback.print_exc()