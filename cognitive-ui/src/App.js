import React, { useState } from 'react';
import { Upload, Activity, Brain, FileText, Mic, Eye, Heart, AlertCircle, CheckCircle } from 'lucide-react';

const CognitiveDeclineDetector = () => {
  const [activeTab, setActiveTab] = useState('upload');
  const [files, setFiles] = useState({
    speech: null,
    text: null,
    handwriting: null,
    visual: null,
    physiological: null
  });
  const [results, setResults] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const modalityInfo = {
    speech: {
      icon: Mic,
      title: 'Speech Analysis',
      description: 'Audio recording (WAV/MP3)',
      features: ['Pause frequency', 'Speech rate', 'Articulation', 'Fluency']
    },
    text: {
      icon: FileText,
      title: 'Text Analysis',
      description: 'Written text sample (TXT)',
      features: ['Vocabulary complexity', 'Grammar', 'Coherence', 'Repetition']
    },
    handwriting: {
      icon: FileText,
      title: 'Handwriting Analysis',
      description: 'Handwriting image (PNG/JPG)',
      features: ['Tremor patterns', 'Pressure', 'Spacing', 'Letter formation']
    },
    visual: {
      icon: Eye,
      title: 'Visual Test Results',
      description: 'Visual cognition test (JSON)',
      features: ['Reaction time', 'Pattern recognition', 'Memory recall', 'Attention span']
    },
    physiological: {
      icon: Heart,
      title: 'Physiological Data',
      description: 'Health metrics (CSV)',
      features: ['Heart rate variability', 'Sleep patterns', 'Activity levels', 'Vital signs']
    }
  };

  const handleFileUpload = (modality, event) => {
    const file = event.target.files[0];
    if (file) {
      setFiles(prev => ({ ...prev, [modality]: file }));
    }
  };

  const analyzeData = async () => {
    setIsAnalyzing(true);
    
    // Simulate ML model inference
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    // Generate mock results based on uploaded files
    const uploadedCount = Object.values(files).filter(f => f !== null).length;
    const confidence = 0.65 + (uploadedCount * 0.05);
    const hasDecline = Math.random() > 0.5;
    
    const mockResults = {
      prediction: hasDecline ? 'Cognitive Decline Detected' : 'No Significant Decline',
      confidence: (confidence * 100).toFixed(1),
      riskLevel: hasDecline ? 'Moderate' : 'Low',
      modalityScores: {
        speech: files.speech ? (0.3 + Math.random() * 0.5).toFixed(2) : null,
        text: files.text ? (0.4 + Math.random() * 0.4).toFixed(2) : null,
        handwriting: files.handwriting ? (0.35 + Math.random() * 0.45).toFixed(2) : null,
        visual: files.visual ? (0.25 + Math.random() * 0.55).toFixed(2) : null,
        physiological: files.physiological ? (0.3 + Math.random() * 0.5).toFixed(2) : null
      },
      keyFindings: [
        {
          modality: 'Speech',
          finding: files.speech ? 'Increased pause duration between words (avg: 1.2s vs normal 0.5s)' : 'Not analyzed',
          severity: files.speech ? (hasDecline ? 'high' : 'low') : 'none'
        },
        {
          modality: 'Text',
          finding: files.text ? 'Vocabulary diversity score: 0.72 (normal range: 0.75-0.95)' : 'Not analyzed',
          severity: files.text ? (hasDecline ? 'medium' : 'low') : 'none'
        },
        {
          modality: 'Handwriting',
          finding: files.handwriting ? 'Minor tremor detected in letter formation' : 'Not analyzed',
          severity: files.handwriting ? (hasDecline ? 'medium' : 'low') : 'none'
        },
        {
          modality: 'Visual',
          finding: files.visual ? 'Reaction time: 450ms (normal: <400ms)' : 'Not analyzed',
          severity: files.visual ? (hasDecline ? 'medium' : 'low') : 'none'
        },
        {
          modality: 'Physiological',
          finding: files.physiological ? 'Sleep efficiency: 78% (optimal: >85%)' : 'Not analyzed',
          severity: files.physiological ? (hasDecline ? 'medium' : 'low') : 'none'
        }
      ],
      explanation: hasDecline 
        ? 'The multimodal analysis indicates patterns consistent with mild cognitive decline. Multiple modalities show deviations from normal baselines, particularly in speech fluency, reaction time, and executive function markers. Early intervention and regular monitoring are recommended.'
        : 'The analysis shows cognitive performance within normal ranges across all tested modalities. No significant indicators of cognitive decline were detected. Continue regular health monitoring and maintain cognitive engagement activities.'
    };
    
    setResults(mockResults);
    setIsAnalyzing(false);
  };

  const resetAnalysis = () => {
    setFiles({
      speech: null,
      text: null,
      handwriting: null,
      visual: null,
      physiological: null
    });
    setResults(null);
    setActiveTab('upload');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 p-6">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="bg-white rounded-2xl shadow-xl p-8 mb-6">
          <div className="flex items-center gap-4 mb-4">
            <div className="bg-gradient-to-br from-blue-500 to-purple-600 p-3 rounded-xl">
              <Brain className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-gray-800">AI Cognitive Decline Detection</h1>
              <p className="text-gray-600">Multimodal Analysis System</p>
            </div>
          </div>
          <div className="bg-blue-50 border-l-4 border-blue-500 p-4 rounded">
            <p className="text-sm text-gray-700">
              This system analyzes multiple data modalities using deep learning to detect early signs of cognitive decline. Upload data from various sources for comprehensive assessment.
            </p>
          </div>
        </div>

        {/* Main Content */}
        {!results ? (
          <div className="bg-white rounded-2xl shadow-xl p-8">
            {/* Tab Navigation */}
            <div className="flex gap-4 mb-6 border-b">
              <button
                onClick={() => setActiveTab('upload')}
                className={`pb-3 px-4 font-semibold transition-colors ${
                  activeTab === 'upload'
                    ? 'border-b-2 border-blue-500 text-blue-600'
                    : 'text-gray-500 hover:text-gray-700'
                }`}
              >
                <Upload className="w-5 h-5 inline mr-2" />
                Upload Data
              </button>
              <button
                onClick={() => setActiveTab('info')}
                className={`pb-3 px-4 font-semibold transition-colors ${
                  activeTab === 'info'
                    ? 'border-b-2 border-blue-500 text-blue-600'
                    : 'text-gray-500 hover:text-gray-700'
                }`}
              >
                <Activity className="w-5 h-5 inline mr-2" />
                About Analysis
              </button>
            </div>

            {/* Upload Tab */}
            {activeTab === 'upload' && (
              <div>
                <div className="grid md:grid-cols-2 gap-6 mb-8">
                  {Object.entries(modalityInfo).map(([key, info]) => {
                    const Icon = info.icon;
                    return (
                      <div
                        key={key}
                        className="border-2 border-dashed border-gray-300 rounded-xl p-6 hover:border-blue-400 transition-colors"
                      >
                        <div className="flex items-start gap-4">
                          <div className="bg-blue-100 p-3 rounded-lg">
                            <Icon className="w-6 h-6 text-blue-600" />
                          </div>
                          <div className="flex-1">
                            <h3 className="font-semibold text-gray-800 mb-1">{info.title}</h3>
                            <p className="text-sm text-gray-600 mb-3">{info.description}</p>
                            
                            <label className="cursor-pointer">
                              <input
                                type="file"
                                onChange={(e) => handleFileUpload(key, e)}
                                className="hidden"
                                accept={
                                  key === 'speech' ? 'audio/*' :
                                  key === 'handwriting' ? 'image/*' :
                                  key === 'physiological' ? '.csv' :
                                  key === 'visual' ? '.json' : '.txt'
                                }
                              />
                              <div className="bg-blue-50 hover:bg-blue-100 text-blue-600 px-4 py-2 rounded-lg text-sm font-medium transition-colors inline-block">
                                {files[key] ? '✓ Uploaded' : 'Choose File'}
                              </div>
                            </label>
                            
                            {files[key] && (
                              <p className="text-xs text-green-600 mt-2">
                                {files[key].name}
                              </p>
                            )}
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>

                <button
                  onClick={analyzeData}
                  disabled={Object.values(files).every(f => f === null) || isAnalyzing}
                  className="w-full bg-gradient-to-r from-blue-500 to-purple-600 text-white py-4 rounded-xl font-semibold text-lg hover:shadow-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isAnalyzing ? (
                    <span className="flex items-center justify-center gap-2">
                      <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                      Analyzing Data...
                    </span>
                  ) : (
                    'Analyze Cognitive Status'
                  )}
                </button>
              </div>
            )}

            {/* Info Tab */}
            {activeTab === 'info' && (
              <div className="space-y-6">
                <div>
                  <h3 className="text-xl font-bold text-gray-800 mb-4">How It Works</h3>
                  <p className="text-gray-700 mb-4">
                    Our system uses a multimodal deep learning approach to detect cognitive decline patterns across five key modalities:
                  </p>
                </div>

                {Object.entries(modalityInfo).map(([key, info]) => {
                  const Icon = info.icon;
                  return (
                    <div key={key} className="bg-gray-50 rounded-xl p-6">
                      <div className="flex items-start gap-4">
                        <div className="bg-blue-100 p-3 rounded-lg">
                          <Icon className="w-6 h-6 text-blue-600" />
                        </div>
                        <div>
                          <h4 className="font-semibold text-gray-800 mb-2">{info.title}</h4>
                          <p className="text-sm text-gray-600 mb-3">{info.description}</p>
                          <div className="text-sm text-gray-700">
                            <strong>Key Features Analyzed:</strong>
                            <ul className="list-disc list-inside mt-2 space-y-1">
                              {info.features.map((feature, idx) => (
                                <li key={idx}>{feature}</li>
                              ))}
                            </ul>
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })}

                <div className="bg-yellow-50 border-l-4 border-yellow-500 p-4 rounded">
                  <p className="text-sm text-gray-700">
                    <strong>Note:</strong> This is a screening tool and should not replace professional medical diagnosis. Always consult healthcare providers for clinical assessment.
                  </p>
                </div>
              </div>
            )}
          </div>
        ) : (
          /* Results View */
          <div className="space-y-6">
            {/* Overall Result */}
            <div className={`rounded-2xl shadow-xl p-8 ${
              results.prediction.includes('Decline') 
                ? 'bg-gradient-to-br from-orange-50 to-red-50 border-2 border-orange-200' 
                : 'bg-gradient-to-br from-green-50 to-emerald-50 border-2 border-green-200'
            }`}>
              <div className="flex items-start gap-4 mb-6">
                {results.prediction.includes('Decline') ? (
                  <AlertCircle className="w-12 h-12 text-orange-600" />
                ) : (
                  <CheckCircle className="w-12 h-12 text-green-600" />
                )}
                <div>
                  <h2 className="text-2xl font-bold text-gray-800 mb-2">
                    {results.prediction}
                  </h2>
                  <p className="text-gray-700">
                    Confidence: <strong>{results.confidence}%</strong> | Risk Level: <strong>{results.riskLevel}</strong>
                  </p>
                </div>
              </div>

              <div className="bg-white bg-opacity-70 rounded-xl p-6">
                <h3 className="font-semibold text-gray-800 mb-3">Clinical Interpretation:</h3>
                <p className="text-gray-700 leading-relaxed">{results.explanation}</p>
              </div>
            </div>

            {/* Modality Scores */}
            <div className="bg-white rounded-2xl shadow-xl p-8">
              <h3 className="text-xl font-bold text-gray-800 mb-6">Modality Analysis Scores</h3>
              <div className="space-y-4">
                {Object.entries(results.modalityScores).map(([modality, score]) => {
                  if (!score) return null;
                  const percentage = (parseFloat(score) * 100).toFixed(0);
                  const Icon = modalityInfo[modality].icon;
                  
                  return (
                    <div key={modality} className="space-y-2">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <Icon className="w-5 h-5 text-blue-600" />
                          <span className="font-medium text-gray-700 capitalize">{modality}</span>
                        </div>
                        <span className="text-sm font-semibold text-gray-600">{percentage}%</span>
                      </div>
                      <div className="bg-gray-200 rounded-full h-3 overflow-hidden">
                        <div
                          className={`h-full rounded-full transition-all ${
                            percentage > 70 ? 'bg-red-500' :
                            percentage > 50 ? 'bg-yellow-500' : 'bg-green-500'
                          }`}
                          style={{ width: `${percentage}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Key Findings */}
            <div className="bg-white rounded-2xl shadow-xl p-8">
              <h3 className="text-xl font-bold text-gray-800 mb-6">Detailed Findings</h3>
              <div className="space-y-4">
                {results.keyFindings.map((finding, idx) => (
                  <div
                    key={idx}
                    className={`p-4 rounded-xl border-l-4 ${
                      finding.severity === 'high' ? 'bg-red-50 border-red-500' :
                      finding.severity === 'medium' ? 'bg-yellow-50 border-yellow-500' :
                      finding.severity === 'low' ? 'bg-green-50 border-green-500' :
                      'bg-gray-50 border-gray-300'
                    }`}
                  >
                    <div className="flex items-start justify-between">
                      <div>
                        <h4 className="font-semibold text-gray-800 mb-1">{finding.modality}</h4>
                        <p className="text-sm text-gray-700">{finding.finding}</p>
                      </div>
                      {finding.severity !== 'none' && (
                        <span className={`text-xs font-semibold px-3 py-1 rounded-full ${
                          finding.severity === 'high' ? 'bg-red-200 text-red-800' :
                          finding.severity === 'medium' ? 'bg-yellow-200 text-yellow-800' :
                          'bg-green-200 text-green-800'
                        }`}>
                          {finding.severity.toUpperCase()}
                        </span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Actions */}
            <div className="bg-white rounded-2xl shadow-xl p-8">
              <h3 className="text-xl font-bold text-gray-800 mb-4">Recommended Actions</h3>
              <ul className="space-y-3 mb-6">
                <li className="flex items-start gap-3">
                  <CheckCircle className="w-5 h-5 text-green-600 mt-0.5 flex-shrink-0" />
                  <span className="text-gray-700">Consult with a neurologist or geriatrician for comprehensive evaluation</span>
                </li>
                <li className="flex items-start gap-3">
                  <CheckCircle className="w-5 h-5 text-green-600 mt-0.5 flex-shrink-0" />
                  <span className="text-gray-700">Consider neuropsychological testing for detailed cognitive assessment</span>
                </li>
                <li className="flex items-start gap-3">
                  <CheckCircle className="w-5 h-5 text-green-600 mt-0.5 flex-shrink-0" />
                  <span className="text-gray-700">Maintain regular monitoring with follow-up assessments every 3-6 months</span>
                </li>
                <li className="flex items-start gap-3">
                  <CheckCircle className="w-5 h-5 text-green-600 mt-0.5 flex-shrink-0" />
                  <span className="text-gray-700">Engage in cognitive stimulation activities and maintain social connections</span>
                </li>
              </ul>

              <button
                onClick={resetAnalysis}
                className="w-full bg-gradient-to-r from-blue-500 to-purple-600 text-white py-4 rounded-xl font-semibold text-lg hover:shadow-lg transition-all"
              >
                Start New Analysis
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default CognitiveDeclineDetector;
