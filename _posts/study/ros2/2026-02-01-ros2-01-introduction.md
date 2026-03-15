---
title: "[ROS2] 01: Introduction (ROS2 소개)"
categories: [ROS2]
tags: [ROS2, Robotics, ROS, Robot Operating System]
article_header:
  type: overlay
  theme: dark
  background_color: '#0d1b2a'
  background_image:
    gradient: 'linear-gradient(135deg, rgba(13, 27, 42, .85), rgba(90, 150, 120, .65))'
    src: /assets/images/study/deep-learning.jpg
mathjax: true
mathjax_autoNumber: true
---

**ROS2 (Robot Operating System 2) 학습 시리즈**

ROS2에 대한 기본적인 내용에 대해서 정리하고 설명하고자 한다. 

<!--more-->

## 1. Introduction

**ROS2 (Robot Operating System 2)**는 로봇 소프트웨어 개발을 위한 오픈소스 프레임워크입니다. 원래 ROS1의 후속 버전으로 개발되었으며, **실시간 성능**, **DDS(Data Distribution Service) 기반 통신**, **멀티 플랫폼 지원** 등의 개선사항을 포함시킴

## 2. ROS2의 주요 특징

### 2.0 ROS 1 vs ROS 2 비교 요약

#### 표 1

| Features | ROS 1 | ROS 2 |
| --- | --- | --- |
| Platforms | Linux, macOS | Linux, macOS, Windows |
| Real-time | External frameworks like OROCOS | Real-time nodes when using a proper RTOS with carefully written user code |
| Security | SROS | SROS 2, DDS-Security, Robotic Systems Threat Model |
| Communication | XMLRPC + TCPROS | DDS (RTPS) |
| Middleware interface | - | rmw |
| Node manager (discovery) | ROS Master | No, use DDS's dynamic discovery |
| Languages | C++03, Python 2.7 | C++14 (C++17), Python 3.5+ |
| Client library | roscpp, rospy, rosjava, rosnodejs, and more | rclcpp, rclpy, rcljava, rcljs, and more |
| Build system | rosbuild → catkin (CMake) | ament (CMake), Python setuptools (full support) |
| Build tool | catkin_make, catkin_tools | colcon |

#### 표 2

| Features | ROS 1 | ROS 2 |
| --- | --- | --- |
| Build options | - | Multiple workspace, no non-isolated build, no devel space |
| Version control system | rosws → wstool, rosinstall (*.rosinstall) | vcs tool (*.repos) |
| Life cycle | - | Node life cycle |
| Multiple nodes | One node in a process | Multiple nodes in a process |
| Threading model | Single-threaded or multi-threaded execution | Custom executors |
| Messages (topic, service, action) | *.msg, *.srv, *.action | *.msg, *.srv, *.action, *.idl |
| Command line interface | rosrun, roslaunch, rostopic ... | ros2 run, ros2 launch, ros2 topic ... |
| roslaunch | XML | Python, XML, YAML |
| Graph API | Remapping at startup time only | Remapping at runtime |
| Embedded systems | rosserial, mROS | microROS, XEL Network, ros2arduino, Renesas DDS-XRCE (Micro-XRCE-DDS), AWS ARCLM |

### 2.1 아키텍처

ROS2는 **분산 시스템** 아키텍처를 따릅니다. 각 노드(Node)는 독립적으로 실행되며, **토픽(Topic)**, **서비스(Service)**, **액션(Action)** 을 통해 통신합니다.

### 2.2 DDS 통신 계층

ROS1과 달리 ROS2는 **DDS (Data Distribution Service)** 미들웨어를 사용합니다. 이를 통해:
- 더욱 견고한 통신
- 다양한 QoS (Quality of Service) 옵션
- 보안 기능 내장

### 2.3 지원 플랫폼

- Linux (Ubuntu 22.04 LTS 권장)
- macOS
- Windows 10/11

## 3. 기본 개념

### 3.1 Node (노드)

노드는 ROS2에서 실행되는 최소 단위입니다. 하나의 노드는 특정 기능을 수행하며, 다른 노드들과 통신합니다.

### 3.2 Topic (토픽)

토픽은 **발행자(Publisher)** 와 **구독자(Subscriber)** 간의 비동기 메시지 통신을 위한 채널입니다.

### 3.3 Service (서비스)

서비스는 **요청-응답** 패턴의 동기 통신을 제공합니다. 클라이언트가 요청을 보내면 서버가 응답을 반환합니다.

## 4. 설치 및 환경 설정

### 4.1 Ubuntu 설치 (Humble Hawksbill 예시)

```bash
sudo apt update
sudo apt install ros-humble-desktop
source /opt/ros/humble/setup.bash
```

### 4.2 작업공간 생성

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws
colcon build
source install/setup.bash
```

## 5. 다음 단계

- [02: 노드와 토픽 (Nodes and Topics)](#) - 기본 통신 구조 학습
- [03: 패키지 생성 (Creating Packages)](#) - ROS2 패키지 작성법
- [04: Launch 파일 (Launch Files)](#) - 멀티 노드 실행

---

## 참고 자료

- [ROS2 Official Documentation](https://docs.ros.org/en/humble/)
- [ROS2 Tutorials](https://docs.ros.org/en/humble/Tutorials.html)
