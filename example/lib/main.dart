import 'dart:io';
import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:pose_detection_tflite/pose_detection_tflite.dart';

void main() {
  runApp(const PoseDetectionApp());
}

class PoseDetectionApp extends StatelessWidget {
  const PoseDetectionApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Pose Detection Demo',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        useMaterial3: true,
      ),
      home: const PoseDetectionScreen(),
    );
  }
}

class PoseDetectionScreen extends StatefulWidget {
  const PoseDetectionScreen({super.key});

  @override
  State<PoseDetectionScreen> createState() => _PoseDetectionScreenState();
}

class _PoseDetectionScreenState extends State<PoseDetectionScreen> {
  final PoseDetector _poseDetector = PoseDetector();
  final ImagePicker _picker = ImagePicker();

  bool _isInitialized = false;
  bool _isProcessing = false;
  File? _imageFile;
  PoseDetectionResult? _poseResult;
  String? _errorMessage;

  @override
  void initState() {
    super.initState();
    _initializeDetector();
  }

  Future<void> _initializeDetector() async {
    setState(() {
      _isProcessing = true;
      _errorMessage = null;
    });

    try {
      await _poseDetector.initialize(
        complexity: PoseModelComplexity.lite,
      );

      setState(() {
        _isInitialized = true;
        _isProcessing = false;
      });
    } catch (e) {
      setState(() {
        _isProcessing = false;
        _errorMessage = 'Failed to initialize: $e';
      });
    }
  }

  Future<void> _pickImage(ImageSource source) async {
    try {
      final XFile? pickedFile = await _picker.pickImage(source: source);

      if (pickedFile == null) return;

      setState(() {
        _imageFile = File(pickedFile.path);
        _poseResult = null;
        _isProcessing = true;
        _errorMessage = null;
      });

      // Detect pose
      final result = await _poseDetector.detectPose(_imageFile!);

      print(result);

      setState(() {
        _poseResult = result;
        _isProcessing = false;
        if (result == null) {
          _errorMessage = 'No pose detected in image';
        }
      });
    } catch (e) {
      setState(() {
        _isProcessing = false;
        _errorMessage = 'Error: $e';
      });
    }
  }

  void _showImageSourceDialog() {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: const Text('Select Image Source'),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              ListTile(
                leading: const Icon(Icons.photo_library),
                title: const Text('Gallery'),
                onTap: () {
                  Navigator.pop(context);
                  _pickImage(ImageSource.gallery);
                },
              ),
              ListTile(
                leading: const Icon(Icons.camera_alt),
                title: const Text('Camera'),
                onTap: () {
                  Navigator.pop(context);
                  _pickImage(ImageSource.camera);
                },
              ),
            ],
          ),
        );
      },
    );
  }

  @override
  void dispose() {
    _poseDetector.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Pose Detection Demo'),
        actions: [
          if (_isInitialized && _imageFile != null)
            IconButton(
              icon: const Icon(Icons.info_outline),
              onPressed: _showPoseInfo,
            ),
        ],
      ),
      body: _buildBody(),
      floatingActionButton: _isInitialized && !_isProcessing
          ? FloatingActionButton.extended(
        onPressed: _showImageSourceDialog,
        icon: const Icon(Icons.add_photo_alternate),
        label: const Text('Select Image'),
      )
          : null,
    );
  }

  Widget _buildBody() {
    if (!_isInitialized && _isProcessing) {
      return const Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            CircularProgressIndicator(),
            SizedBox(height: 16),
            Text('Initializing pose detector...'),
          ],
        ),
      );
    }

    if (_errorMessage != null && _imageFile == null) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.error_outline, size: 64, color: Colors.red),
            const SizedBox(height: 16),
            Text(
              _errorMessage!,
              textAlign: TextAlign.center,
              style: const TextStyle(color: Colors.red),
            ),
            const SizedBox(height: 16),
            ElevatedButton(
              onPressed: _initializeDetector,
              child: const Text('Retry'),
            ),
          ],
        ),
      );
    }

    if (_imageFile == null) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              Icons.person_outline,
              size: 100,
              color: Colors.grey[400],
            ),
            const SizedBox(height: 24),
            Text(
              'Select an image to detect pose',
              style: TextStyle(
                fontSize: 18,
                color: Colors.grey[600],
              ),
            ),
            const SizedBox(height: 16),
            ElevatedButton.icon(
              onPressed: _showImageSourceDialog,
              icon: const Icon(Icons.add_photo_alternate),
              label: const Text('Select Image'),
            ),
          ],
        ),
      );
    }

    return SingleChildScrollView(
      child: Column(
        children: [
          // Image with pose overlay
          if (_poseResult != null)
            PoseVisualizerWidget(
              imageFile: _imageFile!,
              poseResult: _poseResult!,
            )
          else
            Image.file(_imageFile!),

          // Processing indicator
          if (_isProcessing)
            const Padding(
              padding: EdgeInsets.all(16.0),
              child: Column(
                children: [
                  CircularProgressIndicator(),
                  SizedBox(height: 8),
                  Text('Detecting pose...'),
                ],
              ),
            ),

          // Error message
          if (_errorMessage != null && !_isProcessing)
            Padding(
              padding: const EdgeInsets.all(16.0),
              child: Card(
                color: Colors.red[50],
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Row(
                    children: [
                      const Icon(Icons.error_outline, color: Colors.red),
                      const SizedBox(width: 8),
                      Expanded(child: Text(_errorMessage!)),
                    ],
                  ),
                ),
              ),
            ),

          // Quick stats
          if (_poseResult != null)
            Padding(
              padding: const EdgeInsets.all(16.0),
              child: Card(
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Pose Detected! ✓',
                        style: Theme.of(context).textTheme.titleLarge?.copyWith(
                          color: Colors.green,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(height: 12),
                      _buildStatRow(
                        'Confidence',
                        '${((1.0 / (1.0 + math.exp(-_poseResult!.detection.score))) * 100).toStringAsFixed(1)}%',
                      ),
                      _buildStatRow(
                        'Landmarks',
                        '${_poseResult!.landmarks.length}',
                      ),
                      _buildStatRow(
                        'Visibility',
                        _poseResult!.isVisible ? 'Good' : 'Poor',
                      ),
                    ],
                  ),
                ),
              ),
            ),

        ],
      ),
    );
  }

  Widget _buildStatRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4.0),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: const TextStyle(fontWeight: FontWeight.w500)),
          Text(value, style: const TextStyle(color: Colors.blue)),
        ],
      ),
    );
  }

  void _showPoseInfo() {
    if (_poseResult == null) return;

    showModalBottomSheet(
      context: context,
      builder: (context) => DraggableScrollableSheet(
        initialChildSize: 0.7,
        minChildSize: 0.5,
        maxChildSize: 0.95,
        expand: false,
        builder: (context, scrollController) => ListView(
          controller: scrollController,
          padding: const EdgeInsets.all(16),
          children: [
            Text(
              'Landmark Details',
              style: Theme.of(context).textTheme.headlineSmall,
            ),
            const SizedBox(height: 16),
            ..._buildLandmarkList(),
          ],
        ),
      ),
    );
  }

  List<Widget> _buildLandmarkList() {
    if (_poseResult == null) return [];

    return _poseResult!.landmarks.map((landmark) {
      final pixel = landmark.toPixel(
        _poseResult!.imageWidth,
        _poseResult!.imageHeight,
      );

      return Card(
        margin: const EdgeInsets.only(bottom: 8),
        child: ListTile(
          leading: CircleAvatar(
            backgroundColor: landmark.visibility > 0.5
                ? Colors.green
                : Colors.orange,
            child: Text(
              landmark.type.index.toString(),
              style: const TextStyle(fontSize: 12),
            ),
          ),
          title: Text(
            _landmarkName(landmark.type),
            style: const TextStyle(fontWeight: FontWeight.w500),
          ),
          subtitle: Text(
            'Position: (${pixel.x}, ${pixel.y})\n'
                'Visibility: ${(landmark.visibility * 100).toStringAsFixed(0)}%',
          ),
          isThreeLine: true,
        ),
      );
    }).toList();
  }

  String _landmarkName(PoseLandmarkType type) {
    return type.toString().split('.').last
        .replaceAllMapped(
      RegExp(r'[A-Z]'),
          (match) => ' ${match.group(0)}',
    )
        .trim();
  }
}

// ============================================================================
// POSE VISUALIZER WIDGET WITH CUSTOM PAINTER
// ============================================================================

class PoseVisualizerWidget extends StatelessWidget {
  final File imageFile;
  final PoseDetectionResult poseResult;

  const PoseVisualizerWidget({
    super.key,
    required this.imageFile,
    required this.poseResult,
  });

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        return Stack(
          children: [
            // Background image
            Image.file(
              imageFile,
              fit: BoxFit.contain,
            ),
            // Pose overlay
            Positioned.fill(
              child: CustomPaint(
                painter: PosePainter(
                  result: poseResult,
                ),
              ),
            ),
          ],
        );
      },
    );
  }
}

// ============================================================================
// CUSTOM PAINTER FOR DRAWING POSE SKELETON
// ============================================================================

class PosePainter extends CustomPainter {
  final PoseDetectionResult result;

  PosePainter({required this.result});

  @override
  void paint(Canvas canvas, Size size) {
    // Calculate scale factors
    final imageAspect = result.imageWidth / result.imageHeight;
    final canvasAspect = size.width / size.height;

    double scaleX, scaleY;
    double offsetX = 0, offsetY = 0;

    if (canvasAspect > imageAspect) {
      // Canvas is wider - fit to height
      scaleY = size.height / result.imageHeight;
      scaleX = scaleY;
      offsetX = (size.width - result.imageWidth * scaleX) / 2;
    } else {
      // Canvas is taller - fit to width
      scaleX = size.width / result.imageWidth;
      scaleY = scaleX;
      offsetY = (size.height - result.imageHeight * scaleY) / 2;
    }

    // Draw connections first (so they appear behind points)
    _drawConnections(canvas, scaleX, scaleY, offsetX, offsetY);

    // Draw landmarks on top
    _drawLandmarks(canvas, scaleX, scaleY, offsetX, offsetY);
  }

  void _drawConnections(Canvas canvas, double scaleX, double scaleY,
      double offsetX, double offsetY) {
    final linePaint = Paint()
      ..color = Colors.green.withOpacity(0.8)
      ..strokeWidth = 3
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;

    // Define skeleton connections
    final connections = [
      // Face
      [PoseLandmarkType.leftEye, PoseLandmarkType.nose],
      [PoseLandmarkType.rightEye, PoseLandmarkType.nose],
      [PoseLandmarkType.leftEye, PoseLandmarkType.leftEar],
      [PoseLandmarkType.rightEye, PoseLandmarkType.rightEar],
      [PoseLandmarkType.mouthLeft, PoseLandmarkType.mouthRight],

      // Torso
      [PoseLandmarkType.leftShoulder, PoseLandmarkType.rightShoulder],
      [PoseLandmarkType.leftShoulder, PoseLandmarkType.leftHip],
      [PoseLandmarkType.rightShoulder, PoseLandmarkType.rightHip],
      [PoseLandmarkType.leftHip, PoseLandmarkType.rightHip],

      // Left arm
      [PoseLandmarkType.leftShoulder, PoseLandmarkType.leftElbow],
      [PoseLandmarkType.leftElbow, PoseLandmarkType.leftWrist],
      [PoseLandmarkType.leftWrist, PoseLandmarkType.leftPinky],
      [PoseLandmarkType.leftWrist, PoseLandmarkType.leftIndex],
      [PoseLandmarkType.leftWrist, PoseLandmarkType.leftThumb],

      // Right arm
      [PoseLandmarkType.rightShoulder, PoseLandmarkType.rightElbow],
      [PoseLandmarkType.rightElbow, PoseLandmarkType.rightWrist],
      [PoseLandmarkType.rightWrist, PoseLandmarkType.rightPinky],
      [PoseLandmarkType.rightWrist, PoseLandmarkType.rightIndex],
      [PoseLandmarkType.rightWrist, PoseLandmarkType.rightThumb],

      // Left leg
      [PoseLandmarkType.leftHip, PoseLandmarkType.leftKnee],
      [PoseLandmarkType.leftKnee, PoseLandmarkType.leftAnkle],
      [PoseLandmarkType.leftAnkle, PoseLandmarkType.leftHeel],
      [PoseLandmarkType.leftAnkle, PoseLandmarkType.leftFootIndex],

      // Right leg
      [PoseLandmarkType.rightHip, PoseLandmarkType.rightKnee],
      [PoseLandmarkType.rightKnee, PoseLandmarkType.rightAnkle],
      [PoseLandmarkType.rightAnkle, PoseLandmarkType.rightHeel],
      [PoseLandmarkType.rightAnkle, PoseLandmarkType.rightFootIndex],
    ];

    for (final connection in connections) {
      final start = result.getLandmark(connection[0]);
      final end = result.getLandmark(connection[1]);

      if (start != null && end != null &&
          start.visibility > 0.5 && end.visibility > 0.5) {
        canvas.drawLine(
          Offset(
            start.x * result.imageWidth * scaleX + offsetX,
            start.y * result.imageHeight * scaleY + offsetY,
          ),
          Offset(
            end.x * result.imageWidth * scaleX + offsetX,
            end.y * result.imageHeight * scaleY + offsetY,
          ),
          linePaint,
        );
      }
    }
  }

  void _drawLandmarks(Canvas canvas, double scaleX, double scaleY,
      double offsetX, double offsetY) {
    for (final landmark in result.landmarks) {
      if (landmark.visibility > 0.5) {
        // Draw point with glow effect
        final center = Offset(
          landmark.x * result.imageWidth * scaleX + offsetX,
          landmark.y * result.imageHeight * scaleY + offsetY,
        );

        // Outer glow
        final glowPaint = Paint()
          ..color = Colors.blue.withOpacity(0.3)
          ..style = PaintingStyle.fill;
        canvas.drawCircle(center, 8, glowPaint);

        // Inner point
        final pointPaint = Paint()
          ..color = Colors.red
          ..style = PaintingStyle.fill;
        canvas.drawCircle(center, 5, pointPaint);

        // White center
        final centerPaint = Paint()
          ..color = Colors.white
          ..style = PaintingStyle.fill;
        canvas.drawCircle(center, 2, centerPaint);
      }
    }
  }

  @override
  bool shouldRepaint(PosePainter oldDelegate) => true;
}